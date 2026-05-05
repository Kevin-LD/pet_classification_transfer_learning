import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.dataset import get_train_val_dataloaders, get_test_dataloader

def imshow(img, ax):
    """
    辅助函数：将 Tensor 展示在指定的 subplot (ax) 上
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = img.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    ax.imshow(img)
    ax.axis('off')

def test_visualization():
    print("正在准备数据加载器...")
    batch_size = 4
    
    # 1. 获取训练集和验证集
    train_loader, val_loader = get_train_val_dataloaders(batch_size=batch_size, num_workers=0)
    # 2. 获取测试集
    test_loader = get_test_dataloader(batch_size=batch_size, num_workers=0)
    
    # 获取类别映射表 (注意：因为使用了 Subset，需要访问 .dataset.dataset)
    # Subset 对象本身没有 classes 属性，它包装在原始 Dataset 里
    class_names = train_loader.dataset.dataset.classes
    
    # 分别获取三个集合的一个批次
    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))
    test_images, test_labels = next(iter(test_loader))

    print("-" * 30)
    print(f"训练集批次形状: {train_images.shape} | 验证集: {val_images.shape} | 测试集: {test_images.shape}")
    
    # 辅助打印函数
    def print_batch_info(name, labels):
        names = [class_names[i] for i in labels]
        print(f"{name} 标签 IDs: {labels.tolist()}")
        print(f"{name} 类别名称: {names}")

    print_batch_info("训练集", train_labels)
    print_batch_info("验证集", val_labels)
    print_batch_info("测试集", test_labels)
    print("-" * 30)

    # 创建 3 行 4 列的画布 (Train, Val, Test)
    fig, axes = plt.subplots(3, batch_size, figsize=(15, 10))
    
    # 第一行：显示训练集 (应包含随机增强效果)
    for i in range(batch_size):
        label_idx = train_labels[i].item()
        imshow(train_images[i], axes[0, i])
        axes[0, i].set_title(f"Train: {class_names[label_idx]}\n(Augmented)", fontsize=9)
    
    # 第二行：显示验证集 (应只有 Resize/Crop，画面较干净)
    for i in range(batch_size):
        label_idx = val_labels[i].item()
        imshow(val_images[i], axes[1, i])
        axes[1, i].set_title(f"Val: {class_names[label_idx]}", fontsize=9)
    
    # 第三行：显示测试集
    for i in range(batch_size):
        label_idx = test_labels[i].item()
        imshow(test_images[i], axes[2, i])
        axes[2, i].set_title(f"Test: {class_names[label_idx]}", fontsize=9)
    
    plt.tight_layout()
    print("可视化完成")
    plt.show()

if __name__ == "__main__":
    test_visualization()
