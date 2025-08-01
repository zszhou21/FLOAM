import os.path
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from data_util import split_data, save_file

num_client = 20
num_task = 10  # You can set this to either 5 or 10
num_classes = 10  # CINIC-10 has 10 classes
alpha = 0.1
np.random.seed(2266)

# 数据集存储地址
datasetroot_dir = "/root/cinic10"
# 生成数据集存储地址
basedir = "./cinic10-incremental-{}-task-{}".format(alpha, num_task)
if not os.path.exists(basedir):
    os.mkdir(basedir)

# Define transforms to apply to the data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                                 std=[0.24205776, 0.23828046, 0.25874835])])

# Map class names to integers
class_to_idx = {
    'airplane': 0,
    'automobile': 1,
    'bird': 2,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'frog': 6,
    'horse': 7,
    'ship': 8,
    'truck': 9
}

def load_cinic10(root, train=True, transform=None):
    dataset_type = 'train' if train else 'valid'
    root = os.path.join(root, dataset_type)
    images = []
    labels = []
    for class_name in os.listdir(root):
        class_path = os.path.join(root, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = Image.open(image_path).convert('RGB')
            if transform:
                image = transform(image)
            images.append(image.numpy())
            labels.append(class_to_idx[class_name])
    return np.array(images), np.array(labels)

# Load the CINIC-10 training dataset
train_images, train_labels = load_cinic10(datasetroot_dir, train=True, transform=transform)

# Load the CINIC-10 test dataset
test_images, test_labels = load_cinic10(datasetroot_dir, train=False, transform=transform)

total_image = np.concatenate((train_images, test_images))
total_label = np.concatenate((train_labels, test_labels))

image_per_client = [[] for _ in range(num_client)]
label_per_client = [[] for _ in range(num_client)]
statistic = [[] for _ in range(num_client)]

# 记录索引的字典 key = client编号  value = []索引list
dataidx_map = {}
# 每一个数据的索引
idxs = np.array(range(len(total_label)))
# 每一个类数据的索引
idx_for_each_class = []
for i in range(num_classes):
    idx_for_each_class.append(idxs[total_label == i])

# 对每类数据操作
for i in range(num_classes):
    num_images = len(idx_for_each_class[i])
    num_per_client = num_images / num_client
    per_client_image_number = [int(num_per_client) for _ in range(num_client)]
    idx = 0
    for client, num_sample in enumerate(per_client_image_number):
        if client not in dataidx_map.keys():
            dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
        else:
            dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample], axis=0)
        idx += num_sample

# 遍历每个客户端,得到每个客户端的索引
df = pd.DataFrame(columns=[str(i) for i in range(10)])
for client in range(num_client):
    idxs = dataidx_map[client]
    image_per_client[client] = total_image[idxs]
    label_per_client[client] = total_label[idxs]
    row = [0 for i in range(10)]
    for i in np.unique(label_per_client[client]):
        statistic[client].append((int(i), int(sum(label_per_client[client] == i))))
        row[i] = int(sum(label_per_client[client] == i))
    df.loc[len(df)] = row
df.to_csv(basedir + "/client-statics.csv")

for client in range(num_client):
    print(f"Client {client}\t Size of data: {len(image_per_client[client])}\t Labels: ", np.unique(label_per_client[client]))
    print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
    print("=" * 50)

K = num_classes
least_samples = len(image_per_client[0]) // 10
if num_task == 10:
    least_samples = len(image_per_client[0]) // 20
print("least samples:", least_samples)
N = least_samples

col = [str(i) for i in range(10)]
col.append('client')
col.append('task')
df = pd.DataFrame(columns=col)

# 将每类数据按照迪利克雷分布分配给每个任务
for client_id in range(num_client):
    client_images = image_per_client[client_id]
    client_dataset_label = label_per_client[client_id]
    X = [[] for _ in range(num_task)]
    Y = [[] for _ in range(num_task)]
    client_idx_map = {}

    # 类增量模式
    classes_per_task = num_classes // num_task
    for task in range(num_task):
        start_class = 0
        end_class = 9
        task_classes = list(range(start_class, end_class + 1))

        min_size = 0
        max_attempts = 1000  # 最大尝试次数
        attempt = 0

        idx_batch = [[] for _ in range(num_task)]
        for k in task_classes:
            idx_k = np.where(client_dataset_label == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_task))
            proportions = np.clip(proportions, a_min=0.05, a_max=None)  # 确保最小比例
            proportions /= proportions.sum()  # 归一化
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_task):
            client_idx_map[j] = idx_batch[j]

    for task_id in range(num_task):
        row = [0 for i in range(12)]
        row[10] = client_id
        row[11] = task_id
        idxs = client_idx_map[task_id]
        Y[task_id] = client_dataset_label[idxs]
        X[task_id] = client_images[idxs]

        info = []
        for i in np.unique(Y[task_id]):
            info.append((int(i), int(sum(Y[task_id] == i))))
            row[i] = int(sum(Y[task_id] == i))
        df.loc[len(df)] = row

        print(f"Client {client_id}  Task {task_id}\t Size of data: {len(X[task_id])}\t Labels: ", np.unique(Y[task_id]))
        print(f"\t\t Samples of labels: ", [i for i in info])
        print("-" * 50)
        print("=" * 50 + "\n\n")

    # 保存数据
    train_data, test_data = split_data(X, Y)

    if not os.path.exists(basedir + "/train"):
        os.mkdir(basedir + "/train")
    if not os.path.exists(basedir + "/test"):
        os.mkdir(basedir + "/test")

    train_path = basedir + "/train/client-" + str(client_id) + "-task-"
    test_path = basedir + "/test/client-" + str(client_id) + "-task-"
    save_file(train_path, test_path, train_data, test_data)

df.to_csv(basedir + "/task-statics.csv")

path = basedir + '/test/'
all_test_data = {}
for client_id in range(num_client):
    for task in range(num_task):
        file = path + 'client-' + str(client_id) + '-task-' + str(task) + '.npz'
        with open(file, 'rb') as f:
            data = np.load(f, allow_pickle=True)['data'].tolist()
            if 'x' not in all_test_data.keys():
                all_test_data['x'] = data['x']
                all_test_data['y'] = data['y']
            else:
                all_test_data['x'] = np.concatenate((all_test_data['x'], data['x']))
                all_test_data['y'] = np.concatenate((all_test_data['y'], data['y']))

test_path = basedir + "/test/test-data"

with open(test_path + '.npz', 'wb') as f:
    np.savez_compressed(f, data=(all_test_data))