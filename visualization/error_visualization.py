import os
import sys
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random  # 导入随机模块

# --- 将项目根目录加入 import 搜索路径 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.dataset import get_test_dataloader
from models.resnet_custom import get_baseline_model
from models.vit_tiny import get_vit_tiny

def denormalize(tensor):
    """
    反归一化图像以便正常显示
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img

def main():
    parser = argparse.ArgumentParser(description="随机模型错例可视化工具")
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--num_samples', type=int, default=8, help='展示错例的数量')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据
    test_loader = get_test_dataloader(root='data', batch_size=args.batch_size, image_size=224)
    try:
        class_names = test_loader.dataset.dataset.classes
    except AttributeError:
        class_names = test_loader.dataset.classes

    # 2. 加载模型逻辑
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"未找到模型: {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location=device)
    use_vit = checkpoint.get('use_vit', False)
    model_name = checkpoint.get('model_name', 'resnet18')
    use_attention = checkpoint.get('use_attention', None)

    if use_vit:
        model = get_vit_tiny(num_classes=len(class_names), pretrained=False)
    else:
        model = get_baseline_model(model_name=model_name, num_classes=len(class_names), 
                                   pretrained=False, use_attention=use_attention)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 3. 寻找所有错例
    all_errors = []
    print(f"正在全量遍历测试集以搜寻错例（共 {len(test_loader.dataset)} 个样本）...")
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            mask = preds != labels
            if mask.any():
                wrong_imgs = inputs[mask].cpu() # 移动到 CPU 节省显存
                wrong_preds = preds[mask].cpu()
                actual_labels = labels[mask].cpu()
                
                for i in range(wrong_imgs.size(0)):
                    all_errors.append({
                        'img': wrong_imgs[i],
                        'pred': class_names[wrong_preds[i]],
                        'label': class_names[actual_labels[i]]
                    })

    # 4. 随机抽样
    if len(all_errors) == 0:
        print("太棒了！测试集中没有发现错例。")
        return

    num_to_show = min(len(all_errors), args.num_samples)
    selected_errors = random.sample(all_errors, num_to_show)
    print(f"发现共 {len(all_errors)} 个错例，随机展示其中的 {num_to_show} 个。")

    # 5. 可视化
    cols = 4
    rows = (num_to_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    
    # 处理只有一行或一列导致 axes 不是数组的情况
    if num_to_show == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(num_to_show):
        err = selected_errors[i]
        img_show = denormalize(err['img'])
        
        axes[i].imshow(img_show)
        axes[i].set_title(f"Pred: {err['pred']}\nTrue: {err['label']}", 
                          color='red', fontsize=9)
        axes[i].axis('off')

    # 隐藏多余的子图
    for j in range(num_to_show, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    
    # 确保 figures 文件夹存在
    save_dir = os.path.join(project_root, "figures")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, "error_example.png")
    plt.savefig(save_path, dpi=200)
    print(f"错例分析图已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
