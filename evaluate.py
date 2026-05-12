import torch
import torch.nn as nn
import argparse
import os
import json

# 1. 导入工具函数
from utils.eval import evaluate
from utils.dataset import get_test_dataloader
from models.resnet_custom import get_baseline_model

def get_model_info(model_path):
    """
    优先级：1. 从 .pth 文件解析 -> 2. 从 config.json 解析
    返回: model_name, use_attention, checkpoint
    """
    model_name = None
    use_attention = None
    checkpoint = None

    # 尝试加载模型文件
    if os.path.exists(model_path):
        try:
            # 仅加载到 CPU 以便解析信息
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                # 尝试从根字典获取
                model_name = checkpoint.get('model_name')
                use_attention = checkpoint.get('use_attention')
                
                # 尝试从保存的 args 字典获取（兼容 train.py 的保存逻辑）
                if 'args' in checkpoint:
                    args_in_ckpt = checkpoint['args']
                    if model_name is None:
                        model_name = args_in_ckpt.get('model_name')
                    if use_attention is None:
                        use_attention = args_in_ckpt.get('use_attention')
                
                if model_name:
                    print(f"[*] 在权重文件中找到模型配置: name={model_name}, attention={use_attention}")
        except Exception as e:
            print(f"警告: 无法解析权重文件中的元数据: {e}")

    # 如果模型文件里没找到，找同目录下的 config.json
    if model_name is None or use_attention is None:
        config_path = os.path.join(os.path.dirname(model_path), 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if model_name is None:
                        model_name = config.get('model_name')
                    if use_attention is None:
                        use_attention = config.get('use_attention')
                    print(f"[*] 在 config.json 中找到配置: name={model_name}, attention={use_attention}")
            except Exception:
                pass

    # 最终兜底
    if model_name is None:
        model_name = "resnet18"
        print(f"[!] 未找到模型名称元数据。使用默认值: name={model_name}, attention={use_attention}")
    
    return model_name, use_attention, checkpoint

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 获取模型详细信息
    model_name, use_attention, checkpoint = get_model_info(args.model_path)

    # 3. 初始化模型结构 (传入解析出的 use_attention)
    print(f"正在构建模型结构: {model_name} (Attention: {use_attention})...")
    model = get_baseline_model(
        model_name=model_name,
        num_classes=37, 
        pretrained=False,
        use_attention=use_attention
    )

    # 4. 加载权重
    if checkpoint is None:
        if not os.path.exists(args.model_path):
            print(f"错误: 未找到模型路径 '{args.model_path}'。")
            return
        checkpoint = torch.load(args.model_path, map_location=device)

    try:
        # 根据保存格式：checkpoint_dict['model_state_dict']
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            # 如果是只有权重的简单 state_dict
            model.load_state_dict(checkpoint, strict=True)
        print(f"成功从 {args.model_path} 加载权重")
    except Exception as e:
        print(f"加载模型权重时出错 (可能是结构不匹配): {e}")
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
    print(f"开始评估模型...")
    loss, acc = evaluate(model, test_loader, criterion, device)

    # 7. 打印结果
    print("\n" + "="*40)
    print(f"评估结果")
    print(f" - 模型名称:   {model_name}")
    print(f" - 注意力机制: {use_attention if use_attention else 'None'}")
    print(f" - 准确率:     {acc * 100:.2f}%")
    print(f" - 损失:       {loss:.4f}")
    print("="*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model on Oxford-IIIT Pet Dataset")
    
    parser.add_argument('--model_path', type=str, required=True, 
                        help='模型路径(.pth)')
    parser.add_argument('--data_root', type=str, default='data', 
                        help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    main(args)
