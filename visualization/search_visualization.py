import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
import sys

def load_search_results(json_path):
    """读取 JSON 并转换为 Pandas DataFrame"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到文件: {json_path}")
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # 将嵌套的字典展平
    records = []
    for item in data:
        record = {'trial': item['trial'], 'val_acc': item.get('best_val_acc', 0.0)}
        record.update(item['search_params'])
        records.append(record)
        
    return pd.DataFrame(records)

def plot_marginal_effects(df, save_dir):
    """绘制每个参数的主效应图"""
    # 排除非参数列
    param_cols = [c for c in df.columns if 
    c not in ['trial', 'val_acc']]
    
    num_params = len(param_cols)
    if num_params == 0:
        return

    # 自适应计算列数 (cols) 和行数 (rows)
    if num_params <= 3:
        cols = num_params
    elif num_params == 4:
        cols = 2
    else:
        cols = 3  # 默认每行 3 个，如果参数非常多也可以改为 4
        
    rows = int(np.ceil(num_params / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()
    
    i = 0
    for i, col in enumerate(param_cols):
        ax = axes[i]
        
        # 判断是连续变量还是离散变量 (根据 unique 值的数量或数据类型)
        is_discrete = df[col].dtype == object or df[col].nunique() <= 5
        
        if is_discrete:
            # 对 lr 和 weight_decay 做统一格式
            if col in ['lr', 'weight_decay']:
                plot_df = df.copy()
                plot_df[col] = plot_df[col].apply(lambda x: f'{x:.1e}')
            else:
                plot_df = df
            sns.boxplot(x=col, y='val_acc', data=plot_df, ax=ax, color='lightgray', showfliers=False)
            sns.stripplot(x=col, y='val_acc', data=plot_df, ax=ax, alpha=0.7, jitter=True)
            ax.set_title(f'Main Effect: {col}')
        else:
            # 连续变量：散点图 + 趋势线
            # 检查是否需要对数坐标 (如果最大最小值跨度超过 2 个数量级)
            use_log_scale = df[col].max() / (df[col].min() + 1e-9) > 100
            
            sns.regplot(x=col, y='val_acc', data=df, ax=ax, 
                        scatter_kws={'alpha':0.6}, line_kws={'color':'red'}, 
                        logx=use_log_scale) # 如果是对数关系，尝试对数回归趋势线
            
            if use_log_scale:
                ax.set_xscale('log')
                ax.set_title(f'Main Effect: {col} (Log Scale)')
            else:
                ax.set_title(f'Main Effect: {col}')
                
        ax.set_ylabel('Validation Accuracy')
        ax.grid(True, linestyle='--', alpha=0.5)

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'marginal_effects.png')
    plt.savefig(save_path, dpi=300)
    print(f"[*] 边缘主效应图已保存至: {save_path}")
    plt.close()

def plot_correlation_heatmap(df, save_dir):
    """绘制参数与准确率之间的相关性热力图 (简单评估哪些参数最重要)"""
    # 仅针对数值型数据计算相关性
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=['trial'], errors='ignore')
    
    plt.figure(figsize=(8, 6))
    corr = numeric_df.corr()
    
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'correlation_heatmap.png')
    plt.savefig(save_path, dpi=300)
    print(f"[*] 相关性热力图已保存至: {save_path}")
    plt.close()

if __name__ == '__main__':
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="分析指定的 search_summary.json 文件并生成可视化图表")
    
    # 输入文件路径
    parser.add_argument(
        '--path', 
        type=str, 
        default="", 
        help='指定 target_json 的完整路径。'
    )
    
    parser.add_argument(
        '--save_dir', 
        type=str, 
        help='指定可视化结果的保存目录。如果不提供，将默认保存在 JSON 文件同级目录下的。'
    )
    
    args = parser.parse_args()
    target_json = args.path

    # 验证输入文件是否存在
    if not os.path.exists(target_json):
        print(f"[!] 错误: 指定的文件不存在 -> {target_json}")
        sys.exit(1)

    # 2. 确定 save_dir
    if args.save_dir:
        save_dir = args.save_dir
    else:
        # 如果没指定，则使用 JSON 文件所在的目录
        save_dir = os.path.join(os.path.dirname(target_json), "exp_figures")
    
    # 如果指定的保存目录不存在，则创建它
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"[*] 已创建保存目录: {save_dir}")

    # 3. 执行可视化
    print(f"[*] 正在分析: {target_json}")
    print(f"[*] 结果将保存至: {save_dir}")
    
    try:
        df = load_search_results(target_json)
        plot_marginal_effects(df, save_dir)
        plot_correlation_heatmap(df, save_dir)
        print("[*] 可视化分析全部完成！")
    except Exception as e:
        print(f"[!] 可视化过程中发生错误: {e}")
