import os
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_loss_only(history_path):
    """
    解析 history.json 并只生成 Loss 训练曲线
    """
    # 1. 检查文件是否存在
    if not os.path.exists(history_path):
        print(f"错误: 找不到历史记录文件 '{history_path}'")
        return

    # 2. 加载数据
    with open(history_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    epoch_data = data.get('epoch_history', [])

    if not epoch_data:
        print("错误: history.json 中没有有效的 epoch 数据。")
        return

    # 3. 数据提取
    epochs = [x['epoch'] for x in epoch_data]
    train_loss = [x['train_loss'] for x in epoch_data]
    val_loss = [x['val_loss'] for x in epoch_data]

    # 4. 创建画布 (单图)
    plt.figure(figsize=(10, 6))
    
    # 绘制曲线
    plt.plot(epochs, train_loss, 'b-o', label='Train Loss', markersize=5, linewidth=2)
    plt.plot(epochs, val_loss, 'r-s', label='Val Loss', markersize=5, linewidth=2)

    # 图表装饰
    plt.title('Training and Validation Loss', fontsize=15, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 强制 X 轴显示整数（Epoch）
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # 5. 自动整理布局并保存
    plt.tight_layout()
    
    # 路径处理
    exp_dir = os.path.dirname(history_path)
    save_dir = os.path.join(exp_dir, 'exp_figures')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n[Success] Loss 曲线已保存至: {save_path}")
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Training Loss only.')
    parser.add_argument('--path', type=str, required=True, 
                        help='history.json 的完整路径')
    
    args = parser.parse_args()
    plot_loss_only(args.path)
