import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

def get_train_val_dataloaders(root='data', batch_size=32, image_size=224, num_workers=4, val_split=0.1):
    """
    获取 Oxford-IIIT Pet 数据集的训练和验证 DataLoader
    root (str): 数据存储根目录
    batch_size (int): 批大小
    image_size (int): 最终输入模型的图片尺寸
    num_workers (int): 加载数据的线程数
    val_split (float): 验证集占 trainval 的比例
    """
    
    # 训练集变换：包含随机增强
    train_transform = transforms.Compose([
        # 直接在原图上进行随机面积比例和长宽比的裁剪，然后 Resize 到目标尺寸
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(), # 随机翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 加入轻微色彩抖动
        transforms.ToTensor(),
        # 使用 ImageNet 的均值和标准差进行归一化
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 验证集变换：仅 Resize 和归一化
    val_transform = transforms.Compose([
        # 先把短边缩放到 256，保持长宽比
        transforms.Resize(256),
        # 从中心切出 224x224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 为了让 train 和 val 使用不同的 transform，我们需要加载两次数据集
    # 它们共享同一个 root 目录，所以不会重复下载
    full_train_ds = datasets.OxfordIIITPet(
        root=root, split='trainval', target_types='category', download=True, transform=train_transform
    )
    full_val_ds = datasets.OxfordIIITPet(
        root=root, split='trainval', target_types='category', download=True, transform=val_transform
    )

    num_total = len(full_train_ds)
    num_val = int(num_total * val_split)
    num_train = num_total - num_val

    # 使用 torch.utils.data.random_split 生成随机索引
    # 固定 generator 以保证实验可复现
    train_indices, val_indices = random_split(
        range(num_total), [num_train, num_val], 
        generator=torch.Generator().manual_seed(42)
    )

    # 创建 Subset
    train_dataset = Subset(full_train_ds, train_indices)
    val_dataset = Subset(full_val_ds, val_indices)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        persistent_workers=(num_workers > 0), 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        persistent_workers=(num_workers > 0), 
        num_workers=num_workers
    )

    return train_loader, val_loader

def get_test_dataloader(root='data', batch_size=32, image_size=224, num_workers=4):
    """
    获取 Oxford-IIIT Pet 数据集的测试 DataLoader
    root (str): 数据存储根目录
    batch_size (int): 批大小
    image_size (int): 最终输入模型的图片尺寸
    num_workers (int): 加载数据的线程数
    """

    test_transform = transforms.Compose([
        # 先把短边缩放到 256，保持长宽比
        transforms.Resize(256),
        # 从中心切出 224x224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载测试集
    test_dataset = datasets.OxfordIIITPet(
        root=root,
        split='test',
        target_types='category',
        download=True,
        transform=test_transform
    )

    # 创建测试集 DataLoader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        persistent_workers=(num_workers > 0), 
        num_workers=num_workers
    )

    return test_loader
