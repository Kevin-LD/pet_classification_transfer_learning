import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_pet_dataloader(root='data', batch_size=32, image_size=224, num_workers=4):
    """
    获取 Oxford-IIIT Pet 数据集的训练和测试 DataLoader
    root (str): 数据存储根目录
    batch_size (int): 批大小
    image_size (int): 最终输入模型的图片尺寸
    num_workers (int): 加载数据的线程数
    """
    
    train_transform = transforms.Compose([
    # 直接在原图上进行随机面积比例和长宽比的裁剪，然后 Resize 到目标尺寸
    # scale=(0.08, 1.0) 表示随机裁剪原图面积的 8% 到 100%
    # ratio=(0.75, 1.33) 表示长宽比在 3/4 到 4/3 之间随机
    transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(), # 随机翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 加入轻微色彩抖动，增加鲁棒性
    transforms.ToTensor(),
    # 使用 ImageNet 的均值和标准差进行归一化，因为我们要用预训练模型
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        # 先把短边缩放到 256，保持长宽比
        transforms.Resize(256),
        # 从中心切出 224x224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = datasets.OxfordIIITPet(
        root=root,
        split='trainval',
        target_types='category',
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.OxfordIIITPet(
        root=root,
        split='test',
        target_types='category',
        download=True,
        transform=test_transform
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    return train_loader, test_loader
