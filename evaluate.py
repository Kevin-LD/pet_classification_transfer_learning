import torch
import torch.nn as nn
import argparse
import os
import json

# 1. 导入你的工具函数
from utils.eval import evaluate
from utils.dataset import get_test_dataloader
from models.baseline import get_baseline_model

def get_model_info(model_path, default_name):
    """
    优先级：1. 从 .pth 文件解析 -> 2. 从 config.json 解析 -> 3. 使用 default_name
    """
    model_name = None
    checkpoint = None

    # 尝试加载模型文件
    if os.path.exists(model_path):
        try:
            # 仅加载到 CPU 以便解析信息
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_name' in checkpoint:
                model_name = checkpoint['model_name']
                print(f"[*] 在权重文件中找到模型名称: {model_name}")
        except Exception as e:
            print(f"警告: 无法解析权重文件中的模型名称: {e}")

    # 如果模型文件里没找到，找 config.json
    if model_name is None:
        config_path = os.path.join(os.path.dirname(model_path), 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    model_name = config.get('model_name')
                    print(f"[*] 在 config.json 中找到模型名称: {model_name}")
            except Exception:
                pass

    # 最终兜底
    if model_name is None:
        model_name = default_name
        print(f"[!] 未找到元数据。使用默认模型名称: {model_name}")

    return model_name, checkpoint

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 获取模型名称和 checkpoint 内容
    model_name, checkpoint = get_model_info(args.model_path, args.model_name)

    # 3. 初始化模型结构
    model = get_baseline_model(
        model_name=model_name,
        num_classes=37, 
        pretrained=False 
    )

    # 4. 加载权重
    if checkpoint is None:
        if not os.path.exists(args.model_path):
            print(f"错误: 未找到模型路径 '{args.model_path}'。")
            return
        checkpoint = torch.load(args.model_path, map_location=device)

    try:
        # 根据你提供的格式：checkpoint_dict['model_state_dict']
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果是只有权重的简单 state_dict
            model.load_state_dict(checkpoint)
        print(f"成功从 {args.model_path} 加载权重")
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        return

    model.to(device)

    # 5. 获取数据加载器
    test_loader = get_test_dataloader(
        root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )

    # 6. 执行评估
    criterion = nn.CrossEntropyLoss()
    print(f"开始评估模型 {model_name}...")
    loss, acc = evaluate(model, test_loader, criterion, device)

    # 7. 打印结果
    print("\n" + "="*40)
    print(f"评估结果")
    print(f" - 模型名称:  {model_name}")
    print(f" - 准确率:    {acc * 100:.2f}%")
    print(f" - 损失:    {loss:.4f}")
    print("="*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model on Oxford-IIIT Pet Dataset")
    
    parser.add_argument('--model_path', type=str, required=True, 
                        help='模型路径(.pth)')
    parser.add_argument('--data_root', type=str, default='data', 
                        help='数据集路径')
    parser.add_argument('--model_name', type=str, default='resnet18', choices=['resnet18', 'resnet34'], 
                        help="模型架构 (注意：如果 model_path 同目录下存在 config.json，此参数将被覆盖)")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    main(args)
