import json
import os
import argparse
import matplotlib.pyplot as plt

def load_history(file_path):
    """从 JSON 文件加载训练历史数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['epoch_history']

def plot_metrics(histories, labels, output_path):
    """
    对比 Train Loss 和 Validation Accuracy
    """
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid') 
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 颜色配置
    colors = ['#1f77b4', '#d62728', '#2ca02c']
    markers = ['o', 's', '^']

    for i, (history, label) in enumerate(zip(histories, labels)):
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        val_acc = [h['val_acc'] for h in history]

        # --- 绘制 Train Loss (左图) ---
        axes[0].plot(epochs, train_loss, label=label, 
                     color=colors[i], marker=markers[i], markersize=5, linewidth=2)
        
        # --- 绘制 Validation Accuracy (右图) ---
        axes[1].plot(epochs, val_acc, label=label, 
                     color=colors[i], marker=markers[i], markersize=5, linewidth=2)

    # --- 左图细节 ---
    axes[0].set_title('Training Loss Convergence', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # --- 右图细节 ---
    axes[1].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    # 自动布局
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=300)
    print(f"图像已保存至: {output_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning Model Training Comparison")
    
    parser.add_argument("--baseline", type=str, required=True, help="Baseline JSON path")
    parser.add_argument("--resnet_se", type=str, required=True, help="ResNet+SE JSON path")
    parser.add_argument("--vit", type=str, required=True, help="ViT JSON path")

    # 路径处理
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    DEFAULT_OUTPUT = os.path.join(project_root, "figures", "simplified_comparison.png")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output path")

    args = parser.parse_args()

    try:
        # 加载数据
        histories = [
            load_history(args.baseline),
            load_history(args.resnet_se),
            load_history(args.vit)
        ]
        labels = [
            "Baseline (ResNet)", 
            "ResNet + SE-Block", 
            "Vision Transformer"
        ]

        # 执行绘图
        plot_metrics(histories, labels, args.output)

    except Exception as e:
        print(f"处理文件时出错: {e}")
