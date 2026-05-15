import os
import sys
import torch
import argparse
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# --- 将项目根目录加入 import 搜索路径 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入你项目中的自定义模块
from utils.dataset import get_test_dataloader
from models.resnet_custom import get_baseline_model
from models.vit_tiny import get_vit_tiny

def total_variation_loss(img):
    """
    计算全变分损失 (Total Variation Loss)，用于平滑图像，减少生成过程中的高频噪声。
    """
    tv_h = torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w

def generate_activation_maximization(model, target_class, device, iterations=100, lr=0.05):
    """
    通过梯度上升优化输入图像，使特定目标类别的激活值最大化。
    """
    # 初始化一张带有微小随机噪声的图像 (符合常规的 ImageNet 输入尺寸)
    img = torch.randn(1, 3, 224, 224, device=device) * 0.05
    img.requires_grad_(True)
    
    optimizer = optim.Adam([img], lr=lr)
    
    for i in range(iterations):
        optimizer.zero_grad()
        logits = model(img)
        
        # 我们希望最大化目标类别的得分，因此这里取负号进行最小化
        class_score = logits[0, target_class]
        
        # 加入正则化项以确保生成的图像在视觉上具有连贯性
        l2_reg = 0.01 * torch.norm(img)
        tv_reg = 0.005 * total_variation_loss(img)
        
        loss = -class_score + l2_reg + tv_reg
        # print(loss)
        loss.backward()
        optimizer.step()
    
    # 截断梯度并转换为 numpy 格式用于可视化
    img_np = img.detach().cpu().squeeze().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    
    # 最小-最大归一化到 [0, 1] 范围，方便 Matplotlib 显示
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    return img_np

def main():
    parser = argparse.ArgumentParser(description="Oxford-IIIT Pet 数据集 Deep Dream (激活最大化) 可视化工具")
    parser.add_argument('--model_path', type=str, required=True, help='模型检查点 (.pth) 的路径')
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    DEFAULT_OUTPUT = os.path.join(project_root, "figures", "deep_dream.png")
    parser.add_argument('--save_path', type=str, default=DEFAULT_OUTPUT, help='可视化的保存路径')
    parser.add_argument('--batch_size', type=int, default=32, help='(仅用于加载数据集获取类别名称)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的计算设备: {device}")

    # 1. 加载数据以获取类别名称
    print("正在加载数据集以获取类别信息...")
    test_loader = get_test_dataloader(root='data', batch_size=args.batch_size, image_size=224)
    try:
        class_names = test_loader.dataset.dataset.classes
    except AttributeError:
        class_names = test_loader.dataset.classes

    # 2. 加载模型逻辑 (使用提供的参考逻辑)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"未找到模型: {args.model_path}")
    
    print("正在实例化并加载模型权重...")
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

    # 3. 挑选几个差异较大的类别进行可视化
    target_classes = {}

    indices = [3]
    for idx in indices:
        if idx < len(class_names):
            target_classes[class_names[idx]] = idx

    print(f"准备可视化的类别: {list(target_classes.keys())}")

    # 4. 执行激活最大化并绘图
    fig, axes = plt.subplots(1, len(target_classes), figsize=(16, 4))
    if len(target_classes) == 1:
        axes = [axes]

    for ax, (class_name, class_idx) in zip(axes, target_classes.items()):
        print(f"正在为类别 [{class_name}] (索引 {class_idx}) 生成 Deep Dream 图像...")
        opt_img = generate_activation_maximization(model, class_idx, device)
        
        ax.imshow(opt_img)
        ax.set_title(f"{class_name}\n(Class {class_idx})")
        ax.axis('off')

    # 5. 保存结果
    save_dir = os.path.dirname(os.path.abspath(args.save_path))
    if not os.path.exists(save_dir) and save_dir != "":
        os.makedirs(save_dir)
        
    plt.tight_layout()
    plt.savefig(args.save_path, bbox_inches='tight', dpi=300)
    print(f"Deep Dream 可视化已完成并保存至: {args.save_path}")

if __name__ == "__main__":
    main()
