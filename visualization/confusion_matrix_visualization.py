import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入自定义模块
from utils.eval import evaluate
from utils.dataset import get_test_dataloader
from models.resnet_custom import get_baseline_model
from models.vit_tiny import get_vit_tiny

def plot_confusion_matrix(model, dataloader, device, classes, save_path='confusion_matrix.png'):
    """
    针对多类别优化的混淆矩阵可视化
    """
    print("正在运行模型评估以收集预测结果...")
    criterion = nn.CrossEntropyLoss()
    _, _, all_preds, all_labels = evaluate(
        model, dataloader, criterion, device, return_preds=True
    )

    y_true = all_labels.numpy()
    y_pred = all_preds.numpy()
    cm = confusion_matrix(y_true, y_pred)
    
    # --- 优化方案 ---
    
    # 1. 大幅增加画布尺寸 (根据类别数动态调整)
    # 37类建议宽度至少 18-20 英寸
    plt.figure(figsize=(20, 16)) 
    
    # 2. 使用更细致的绘图参数
    # annot=True: 显示数字；fmt='d': 整数格式
    # annot_kws={"size": 8}: 缩小单元格内数字的字体，防止数字挤出格子
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 8}) 
    
    plt.title('Confusion Matrix (37 Classes)', fontsize=20, pad=20)
    
    # 3. 旋转 X 轴标签并设置对齐方式
    # rotation=45: 倾斜角度；ha='right': 旋转中心对齐右侧，防止标签中心对准刻度导致重叠
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.ylabel('Actual Label', fontsize=14, labelpad=10)
    plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
    
    # 4. 强制使用紧凑布局，防止标签被切掉
    plt.tight_layout()

    # 保存时使用高 DPI 保证文字清晰
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 
    print(f"成功！混淆矩阵已保存至: {save_path}")
    plt.show()

def main():
    # 1. 命令行参数解析
    parser = argparse.ArgumentParser(description="模型混淆矩阵可视化工具")
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型权重路径 (.pth)')
    parser.add_argument('--batch_size', type=int, default=32, help='评估时的 Batch Size')
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    DEFAULT_OUTPUT = os.path.join(project_root, "figures", "confusion_matrix.png")
    parser.add_argument('--save_name', type=str, default=DEFAULT_OUTPUT, help='保存的文件名')
    args = parser.parse_args()

    # 2. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 3. 加载测试集 DataLoader
    # 这里的参数根据你的 get_test_dataloader 定义
    test_loader = get_test_dataloader(root='data', batch_size=args.batch_size, image_size=224, num_workers=4)
    
    # 4. 获取类别名称
    # 根据你提供的逻辑获取 classes
    try:
        # 对应：dataloader -> DataLoader, dataset -> Subset, dataset -> ImageFolder
        class_names = test_loader.dataset.dataset.classes
    except AttributeError:
        # 备选逻辑：如果没用 Subset 包装，直接获取
        class_names = test_loader.dataset.classes

    # 5. 加载模型
    if os.path.exists(args.model_path):
        print(f"正在从 {args.model_path} 加载 Checkpoint...")
        
        # 加载字典
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # 从 checkpoint 中提取配置信息
        # 这样可以保证加载的模型架构与训练时完全一致
        model_name = checkpoint.get('model_name', 'vit_tiny')
        use_vit = checkpoint.get('use_vit', False)
        use_attention = checkpoint.get('use_attention', False)
        
        
        num_classes = len(class_names)
        
        if use_vit:
            print(f"正在构建 ViT 模型...")
            model = get_vit_tiny(
                num_classes=num_classes,
                pretrained=False  # 加载 checkpoint 时通常不需要加载预训练权重
            )
        else:
            print(f"正在构建 ResNet 基础模型: {model_name}, Attention: {use_attention}")
            model = get_baseline_model(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=False,
                use_attention=use_attention
            )
        
        # --- 加载权重 ---
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"模型加载成功！")
    else:
        raise FileNotFoundError(f"找不到模型文件: {args.model_path}")

    # 6. 执行可视化
    plot_confusion_matrix(model, test_loader, device, class_names, save_path=args.save_name)

if __name__ == "__main__":
    main()
