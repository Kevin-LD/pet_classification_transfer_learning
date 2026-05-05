import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# 路径处理，确保能找到 utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.dataset import get_pet_dataloader

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
    train_loader, test_loader = get_pet_dataloader(batch_size=batch_size, num_workers=0)
    
    # 获取类别映射表 (List of strings)
    # 注意：OxfordIIITPet 对象的 classes 属性包含 37 个类别名
    class_names = train_loader.dataset.classes
    
    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))

    print("-" * 30)
    print(f"训练集批次形状 (Batch Images Shape): {train_images.shape}")
    print(f"训练集标签 (Labels IDs): {train_labels.tolist()}")
    # 映射训练集标签
    train_names = [class_names[i] for i in train_labels]
    print(f"训练集类别名称 (Class Names): {train_names}")
    
    print("-" * 15)
    
    print(f"测试集批次形状 (Batch Images Shape): {test_images.shape}")
    print(f"测试集标签 (Labels IDs): {test_labels.tolist()}")
    # 映射测试集标签
    test_names = [class_names[i] for i in test_labels]
    print(f"测试集类别名称 (Class Names): {test_names}")
    print("-" * 30)

    # 创建 2 行 4 列的画布
    fig, axes = plt.subplots(2, batch_size, figsize=(15, 8))
    
    # 第一行：显示训练集 (带 Augmentation)
    for i in range(batch_size):
        label_idx = train_labels[i].item()
        imshow(train_images[i], axes[0, i])
        axes[0, i].set_title(f"Train: {class_names[label_idx]}\n(Augmented)", fontsize=10)
    
    # 第二行：显示测试集
    for i in range(batch_size):
        label_idx = test_labels[i].item()
        imshow(test_images[i], axes[1, i])
        axes[1, i].set_title(f"Test: {class_names[label_idx]}", fontsize=10)
    
    plt.tight_layout()
    print("可视化完成")
    plt.show()

if __name__ == "__main__":
    test_visualization()
