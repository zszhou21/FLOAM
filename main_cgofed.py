#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import numpy as np
import pandas as pd
import torch

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdateCGoFed
from models.test import test_img, test_img_local, test_img_local_all
import os

import pdb
from collections import defaultdict
from sklearn.decomposition import PCA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 新增

def compute_similarity_matrix(client_matrices, server_matrices, target_dim=128):
    """
    Compute similarity matrix between client and server representation matrices.
    Ensure both matrices have the same shape after dimensionality reduction.
    """
    # GPU PCA helper
    def pca_gpu(matrix, n_components):
        X = torch.from_numpy(matrix).float().to(device)
        X = X - X.mean(0)
        # torch.linalg.svd 支持在 GPU 上跑
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        # 主成分投影
        return (X @ Vh[:n_components].T).cpu().numpy()

    similarity_matrix = {}

    for client_idx, client_matrix in client_matrices.items():
        similarity_matrix[client_idx] = {}

        # Dynamically determine the number of components for PCA
        max_components_client = min(client_matrix.shape[0], client_matrix.shape[1])
        pca_dim_client = min(target_dim, max_components_client)

        # GPU PCA
        client_matrix_reduced = pca_gpu(client_matrix, pca_dim_client)

        for task_idx, server_matrix in server_matrices.items():
            # Dynamically determine the number of components for server matrix
            max_components_server = min(server_matrix.shape[0], server_matrix.shape[1])
            pca_dim_server = min(target_dim, max_components_server)

            # Ensure both matrices have the same feature dimension
            pca_dim = min(pca_dim_client, pca_dim_server)

            # GPU PCA
            client_matrix_final = pca_gpu(client_matrix, pca_dim)
            server_matrix_final = pca_gpu(server_matrix, pca_dim)

            # Align shapes by padding or truncating rows
            max_rows = max(client_matrix_final.shape[0], server_matrix_final.shape[0])
            padded_client = np.zeros((max_rows, pca_dim))
            padded_server = np.zeros((max_rows, pca_dim))

            padded_client[:client_matrix_final.shape[0], :] = client_matrix_final
            padded_server[:server_matrix_final.shape[0], :] = server_matrix_final

            # Compute L2 norm distance between aligned matrices
            similarity = np.linalg.norm(padded_client - padded_server, ord=2)
            similarity_matrix[client_idx][task_idx] = similarity

    return similarity_matrix

def select_relevant_tasks(similarity_matrix, top_k=2):
    """
    Select top-k most relevant tasks based on similarity scores.
    """
    selected_tasks = {}
    for client_idx, similarities in similarity_matrix.items():
        sorted_tasks = sorted(similarities.items(), key=lambda x: x[1])[:top_k]
        selected_tasks[client_idx] = [task for task, _ in sorted_tasks]
    return selected_tasks

def aggregate_representation_matrices(client_matrices):
    """
    Aggregate representation matrices from clients.
    Ensure all matrices have the same shape by padding or truncating.
    """
    # 获取所有矩阵的最大行数和列数
    max_rows = max(matrix.shape[0] for matrix in client_matrices.values())
    max_cols = max(matrix.shape[1] for matrix in client_matrices.values())

    # 对每个矩阵进行填充，使其形状一致
    padded_matrices = []
    for matrix in client_matrices.values():
        padded_matrix = np.zeros((max_rows, max_cols))  # 初始化为零矩阵
        padded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix  # 填充原始数据
        padded_matrices.append(padded_matrix)

    # 计算均值
    aggregated_matrix = np.mean(padded_matrices, axis=0)
    return aggregated_matrix

def apply_cross_task_regularization(w_glob, selected_tasks, basis_vectors):
    """
    Apply cross-task gradient regularization to the global model.
    """
    for client_idx, tasks in selected_tasks.items():
        for task in tasks:
            if task in basis_vectors:
                Mt = basis_vectors[task]
                for k, v in w_glob.items():
                    w_glob[k] -= torch.matmul(torch.matmul(v, Mt), Mt.T)  # Regularization term
    return w_glob

if __name__ == '__main__':
    # parse args
    args = args_parser()
    dataset_path = args.datasetpath
    # Seed
    # torch.manual_seed(args.seed)#seed=1
    # torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # np.random.seed(args.seed)
    task_num = args.task_num
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    base_dir = './save/{}/{}_num{}_C{}_le{}_bs{}_round{}_m{}_lr{}/{}/'.format(
        dataset_path, args.model, args.num_users, args.frac, args.local_ep, args.local_bs, args.epochs, args.momentum, args.lr, args.results_save)
    algo_dir = 'cgofed'
    save_folder = './results/cgofed'
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    if not os.path.exists(os.path.join(base_dir, algo_dir)):
        os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)

    # build a global model
    net_glob = get_model(args)
    net_glob.train()

    # build local models
    net_local_list = []
    for user_idx in range(args.num_users):
        net_local_list.append(copy.deepcopy(net_glob))
    
    # training
    results_save_path = os.path.join(base_dir, algo_dir, 'results.csv')

    loss_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []
    

    # Server-side memory to store historical basis vectors and representation matrices
    server_memory = {
        "basis_vectors": {},  # Historical basis vectors for relax-constrained gradient update
        "representation_matrices": {}  # Representation matrices for cross-task gradient regularization
    }
    for iter in range(args.epochs):
        w_glob = None
        loss_locals = []
        
        # Client Sampling
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # print("Round {}, lr: {:.6f}, {}".format(iter, lr, idxs_users))
        
        task=(iter//10)%task_num#每过10个轮次进行任务切换
        print('Current task: ', task)
        uploaded_representation_matrices = {} 
        historical_basis_vectors = server_memory["basis_vectors"]
        # Local Updates
        for idx in idxs_users:
            #数据集名字，序号
            local = LocalUpdateCGoFed(args=args, dataset=dataset_path, idxs=idx, task = task)
            net_local = copy.deepcopy(net_local_list[idx])
            w_local, loss, task_representation_matrix = local.train(net=net_local.to(args.device), lr=lr, historical_basis_vectors=historical_basis_vectors)
                
            loss_locals.append(copy.deepcopy(loss))

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]
            
            uploaded_representation_matrices[idx] = task_representation_matrix
        
        # Aggregation
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], m)
        
        similarity_matrix = compute_similarity_matrix(uploaded_representation_matrices, server_memory["representation_matrices"])
        selected_tasks = select_relevant_tasks(similarity_matrix, top_k=2)  # Select top-k most relevant tasks
        
        server_memory["representation_matrices"][task] = aggregate_representation_matrices(uploaded_representation_matrices)
        # Broadcast
        update_keys = list(w_glob.keys())
        w_glob = apply_cross_task_regularization(w_glob, selected_tasks, server_memory["basis_vectors"])
        for user_idx in range(args.num_users):
            net_local_list[user_idx].load_state_dict(w_glob, strict=False)
        net_glob.load_state_dict(w_glob, strict=False)
        # if (iter + 1) == 50:
        #     lr = 0.01
        # elif (iter + 1) ==75:
        #     lr = 0.001

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        if (iter + 1) % args.test_freq == 0:
            acc_test, acc_test_var, loss_test = test_img_local_all(net_local_list, args, dataset_test=dataset_path, task=task, return_all=False)
            
            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))
            
            #all_acc, all_loss = test_img(net_glob, datatest=dataset_path, args=args)
            all_acc, all_loss = test_img(net_glob, datatest=dataset_path, args=args, epoch = iter, class_num=args.num_classes, save_folder = save_folder)
            
            print('All Test Data: Average loss: {:.4f}, Accuracy: {:.2f}% '.format(
                all_loss, all_acc))

            if best_acc is None or all_acc > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = all_acc
                best_epoch = iter
                
                best_save_path = os.path.join(base_dir, algo_dir, 'best_model.pt')
                
                torch.save(net_best.state_dict(), best_save_path)
                
#                 for user_idx in range(args.num_users):
#                     best_save_path = os.path.join(base_dir, algo_dir, 'best_local_{}.pt'.format(user_idx))
#                     torch.save(net_local_list[user_idx].state_dict(), best_save_path)

            results.append(np.array([iter,task, loss_avg, loss_test, acc_test, all_acc, best_acc]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch','task', 'loss_avg', 'loss_test', 'acc_test',  'all_acc','best_acc'])
            final_results.to_csv(results_save_path, index=False)

    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))