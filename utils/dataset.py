import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def get_train_val_dataloaders(root='data', batch_size=32, image_size=224, num_workers=4, val_split=0.2):
    """
    获取 Oxford-IIIT Pet 数据集的训练和验证 DataLoader
    root (str): 数据存储根目录
    batch_size (int): 批大小
    image_size (int): 最终输入模型的图片尺寸
    num_workers (int): 加载数据的线程数
    val_split (float): 验证集占 trainval 的比例
    """
    
    # 训练集变换：包含数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 验证集变换：仅 Resize 和归一化，保证结果的确定性
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载原始的 trainval 数据集（先不加 transform）
    full_dataset = datasets.OxfordIIITPet(
        root=root,
        split='trainval',
        target_types='category',
        download=True
    )

    # 计算切分索引
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_split, stratify=full_dataset._labels, random_state=42
    )

    # 注意：由于 Subset 指向同一个 dataset 对象，直接修改 transform 会互相影响
    # 这里我们创建两个独立的 Dataset 实例来彻底避免 transform 冲突
    train_dataset = datasets.OxfordIIITPet(
        root=root, split='trainval', target_types='category', transform=train_transform
    )
    val_dataset = datasets.OxfordIIITPet(
        root=root, split='trainval', target_types='category', transform=val_transform
    )
    
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        persistent_workers=(num_workers > 0),
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        persistent_workers=(num_workers > 0),
        num_workers=num_workers,
        pin_memory=True
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
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.OxfordIIITPet(
        root=root,
        split='test',
        target_types='category',
        download=True,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        persistent_workers=(num_workers > 0),
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader
