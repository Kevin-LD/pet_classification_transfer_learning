import os
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def smooth_curve(points, factor=0.95):
    """
    用于平滑 iteration 级别的曲线，使其趋势更明显。
    factor 越大 (接近1)，平滑力度越大。
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_history(history_path):
    """
    解析 history.json 并生成训练可视化图表
    """
    # 1. 检查文件是否存在
    if not os.path.exists(history_path):
        print(f"错误: 找不到历史记录文件 '{history_path}'")
        return

    # 2. 加载数据
    with open(history_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    epoch_data = data.get('epoch_history', [])
    iter_data = data.get('iter_history', [])

    if not epoch_data:
        print("错误: history.json 中没有有效的 epoch 数据。")
        return

    # 3. 数据提取 (List Comprehension)
    epochs = [x['epoch'] for x in epoch_data]
    train_loss = [x['train_loss'] for x in epoch_data]
    val_loss = [x['val_loss'] for x in epoch_data]
    train_acc = [x['train_acc'] for x in epoch_data]
    val_acc = [x['val_acc'] for x in epoch_data]
    epoch_lr = [x['lr'] for x in epoch_data]

    iter_loss = [x['loss'] for x in iter_data]
    iters = range(1, len(iter_loss) + 1)

    # 4. 创建 2x2 画布
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Training Analysis Report', fontsize=18, fontweight='bold')

    # --- [0, 0] Epoch 级别 Loss 曲线 ---
    axs[0, 0].plot(epochs, train_loss, 'b-o', label='Train Loss', markersize=4)
    axs[0, 0].plot(epochs, val_loss, 'r-s', label='Val Loss', markersize=4)
    axs[0, 0].set_title('Epoch Loss (Train vs Val)', fontsize=13)
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- [0, 1] Epoch 级别 Accuracy 曲线 ---
    axs[0, 1].plot(epochs, train_acc, 'b-o', label='Train Acc', markersize=4)
    axs[0, 1].plot(epochs, val_acc, 'g-^', label='Val Acc', markersize=4)
    axs[0, 1].set_title('Epoch Accuracy (Train vs Val)', fontsize=13)
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    axs[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- [1, 0] Epoch 级别学习率变化 ---
    axs[1, 0].plot(epochs, epoch_lr, 'orange', marker='d', label='LR')
    axs[1, 0].set_title('Learning Rate Schedule', fontsize=13)
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Learning Rate')
    axs[1, 0].set_yscale('log') # 学习率通常用对数坐标观察更直观
    axs[1, 0].legend()
    axs[1, 0].grid(True, which="both", linestyle='--', alpha=0.5)
    axs[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- [1, 1] Iteration 级别 Loss ---
    if iter_loss:
        color_loss = 'tab:blue'
        axs[1, 1].set_title('Iteration Training Loss', fontsize=13)
        axs[1, 1].set_xlabel('Iteration')
        axs[1, 1].set_ylabel('Loss', color=color_loss)
        
        # 原始 Loss (浅色) + 平滑后 Loss (深色)
        l1 = axs[1, 1].plot(iters, iter_loss, color='lightblue', alpha=0.3, label='Iter Loss (Raw)')
        l2 = axs[1, 1].plot(iters, smooth_curve(iter_loss, factor=0.9), color=color_loss, label='Iter Loss (Smoothed)')
        axs[1, 1].tick_params(axis='y', labelcolor=color_loss)


        # 合并图例
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        axs[1, 1].legend(lns, labs, loc='upper right', fontsize='small')
        axs[1, 1].grid(True, linestyle='--', alpha=0.4)
    else:
        axs[1, 1].text(0.5, 0.5, 'No Iteration Data Available', ha='center')

    # 5. 自动整理布局并保存
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 路径处理
    exp_dir = os.path.dirname(history_path)
    save_dir = os.path.join(exp_dir, 'exp_figures')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, 'training_report.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n[Success] 可视化图表已保存至: {save_path}")
    
    plt.show()

if __name__ == '__main__':
    # 获取脚本所在目录的上一级作为根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description='Visualize training history from JSON.')
    parser.add_argument('--path', type=str, required=True, 
                        help='history.json 的完整路径')
    
    args = parser.parse_args()
    
    plot_history(args.path)
