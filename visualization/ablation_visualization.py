import json
import argparse
import matplotlib.pyplot as plt
import os

def load_history(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['epoch_history']

def plot_metrics(histories, labels, save_path="ablation_study.png"):
    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 颜色和样式配置
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    styles = ['-', '--', '-.']

    for i, (history, label) in enumerate(zip(histories, labels)):
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        val_acc = [h['val_acc'] for h in history]

        # 绘制训练损失 (Training Loss)
        ax1.plot(epochs, train_loss, label=f"Train Loss: {label}", 
                 color=colors[i], linestyle=styles[i], linewidth=2)
        
        # 绘制验证集准确率 (Validation Accuracy)
        ax2.plot(epochs, val_acc, label=f"Val Acc: {label}", 
                 color=colors[i], linestyle=styles[i], linewidth=2)

    # 格式化左图：Loss
    ax1.set_title("Training Loss Convergence", fontsize=14)
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    # 格式化右图：Accuracy
    ax2.set_title("Validation Accuracy Comparison", fontsize=14)
    ax2.set_xlabel("Epochs", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_ylim(0, 1.0) # 准确率通常在0-1之间
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Study Visualization for Pre-training")
    
    # 接受三个必要的文件路径
    parser.add_argument("--pretrained", type=str, required=True, help="预训练权重历史 JSON 文件路径")
    parser.add_argument("--random", type=str, required=True, help="随机权重历史 JSON 文件路径")
    parser.add_argument("--random_tuned", type=str, required=True, help="随机权重（调优）历史 JSON 文件路径")

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    DEFAULT_OUTPUT = os.path.join(project_root, "figures", "ablation_results.png")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="输出图像保存路径")

    args = parser.parse_args()

    try:
        # 加载数据
        hist_pre = load_history(args.pretrained)
        hist_rand = load_history(args.random)
        hist_rand_tuned = load_history(args.random_tuned)

        histories = [hist_pre, hist_rand, hist_rand_tuned]
        labels = [
            "Pre-trained", 
            "Random (Same Hyperparams)", 
            "Random (High LR & More Epochs)"
        ]

        # 执行绘图
        plot_metrics(histories, labels, args.output)

    except Exception as e:
        print(f"Error processing files: {e}")
