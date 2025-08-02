### Abstract
Federated continual learning (FCL) aims to enable decentralized agents to incrementally acquire knowledge
from non-stationary and privacy-sensitive data distributions. However, real-world deployments face three
compounding challenges: catastrophic forgetting, representation inconsistency across clients, and unstable lo-
cal adaptation due to data heterogeneity. To address these issues, we propose FLOAM (Federated Learning with
Optimized Anchored Memory), a modular framework that integrates semantic anchor memory, anchor-guided
distillation, and meta-adaptive optimization. FLOAM introduces class-level anchor prototypes that serve as
shared semantic references across clients, enabling both contrastive feature alignment and privacy-preserving
knowledge retention. A dynamic hard negative mining mechanism enhances inter-class discriminability,
while meta-learned loss balancing improves client-side stability under continual updates. We evaluate FLOAM
on multiple benchmarks (CIFAR-10, CIFAR-100, Tiny-ImageNet) under class-incremental and multitask set-
tings, demonstrating superior performance over strong FCL baselines in terms of accuracy, forgetting, and
adaptability. FLOAM provides a scalable and principled solution for continual federated intelligence.
