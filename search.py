import os
import json
import random
import math
import itertools
import argparse
import copy
from datetime import datetime
from argparse import Namespace

# 导入训练入口
from train import run_training

def get_base_args():
    """
    定义基础参数。
    注意：save_history 设置为 False 以确保搜索期间不产生冗余的 json 日志。
    """
    return Namespace(
        # 目录与功能
        save_dir='',             
        save_history=False,
        resume='',               
        
        # 模型与数据
        model_name='resnet18',
        pretrained=True,
        num_workers=4,
        
        # 默认优化项 (若未被搜索空间覆盖则用此默认值)
        optimizer='AdamW',
        lr_backbone=1e-5,
        lr_head=1e-3,
        weight_decay=1e-4,
        batch_size=32,
        
        # 搜索阶段固定配置
        epochs=30,               # 固定搜索 Epochs 为 30
        warmup_epochs=0,         # 暂时关闭 Warmup

        # Scheduler 默认值
        lr_scheduler='CosineAnnealingLR',
        lr_step_size=7,
        lr_gamma=0.1
    )

def run_search(cli_args):
    """
    执行超参数搜索
    """
    search_type = cli_args.type
    num_trials = cli_args.trials

    # 1. 定义搜索空间 (新增了 weight_decay)
    search_space = {
        'lr_head': {
            'range': [1e-4, 5e-2], 
            'scale': 'log', 
            'grid_values': [1e-3, 5e-3, 1e-2]
        },
        'lr_backbone': {
            'range': [1e-6, 1e-4], 
            'scale': 'log', 
            'grid_values': [1e-5, 5e-5]
        },
        'weight_decay': {
            'range': [1e-5, 1e-2], 
            'scale': 'log', 
            'grid_values': [1e-5, 1e-4, 1e-3]
        },
        'optimizer': {
            'grid_values': ['AdamW', 'SGD']
        },
        'batch_size': {
            'grid_values': [32, 64]
        }
    }

    keys = list(search_space.keys())
    combinations = []

    # 2. 生成组合列表
    if search_type == 'grid':
        grid_lists = [search_space[k]['grid_values'] for k in keys]
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*grid_lists)]
        print(f"[*] 启动网格搜索，共计 {len(combinations)} 组参数组合。")
    else:
        print(f"[*] 启动随机搜索，将尝试 {num_trials} 组随机参数。")
        for _ in range(num_trials):
            combo = {}
            for k, config in search_space.items():
                if 'range' in config:
                    low, high = config['range']
                    if config.get('scale') == 'log':
                        val = 10 ** random.uniform(math.log10(low), math.log10(high))
                    else:
                        val = random.uniform(low, high)
                    combo[k] = val
                else:
                    combo[k] = random.choice(config['grid_values'])
            combinations.append(combo)

    # 3. 路径准备
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    search_root = cli_args.search_dir or f'./runs/search_{search_type}_{timestamp}'
    os.makedirs(search_root, exist_ok=True)

    results = []

    # 4. 遍历执行
    for i, params in enumerate(combinations):
        print(f"\n" + "#"*60)
        print(f"Trial [{i+1}/{len(combinations)}]")
        
        # 准备该 Trial 的参数对象
        trial_args = get_base_args()
        for k, v in params.items():
            setattr(trial_args, k, v)

        #  LR 线性放缩 (仅针对 SGD)
        if trial_args.optimizer == 'SGD':
            base_bs = 32
            scaling_factor = trial_args.batch_size / base_bs
            if scaling_factor != 1.0:
                trial_args.lr_head *= scaling_factor
                trial_args.lr_backbone *= scaling_factor
                print(f"[*] SGD 检测: 学习率已根据 BatchSize={trial_args.batch_size} 线性放缩 (x{scaling_factor:.2f})")
        else:
            print(f"[*] {trial_args.optimizer} 检测: 保持原始采样学习率")

        # 设置保存目录
        trial_id = f"trial_{i+1}_{trial_args.optimizer}_bs{trial_args.batch_size}"
        trial_args.save_dir = os.path.join(search_root, trial_id)

        try:
            # 执行训练
            best_acc = run_training(trial_args)
            
            # 保存模型全部参数 
            results.append({
                'trial': i + 1,
                'best_val_acc': best_acc,
                'search_params': params,
                'all_params': copy.deepcopy(vars(trial_args)),
                'status': 'Success'
            })
        except Exception as e:
            print(f"[!] Trial {i+1} 运行失败: {e}")
            results.append({
                'trial': i + 1,
                'best_val_acc': 0.0,
                'search_params': params,
                'all_params': copy.deepcopy(vars(trial_args)),
                'status': f'Failed: {str(e)}'
            })

        # 实时保存汇总
        intermediate_summary = sorted(results, key=lambda x: x['best_val_acc'], reverse=True)
        with open(os.path.join(search_root, 'search_summary.json'), 'w') as f:
            json.dump(intermediate_summary, f, indent=4)

    # 5. 打印 Top 3 结果
    print("\n" + "="*60)
    print("搜索任务完成！结果已汇总至 search_summary.json")
    results = sorted(results, key=lambda x: x['best_val_acc'], reverse=True)
    
    for i in range(min(3, len(results))):
        res = results[i]
        print(f"Top {i+1} | Acc: {res['best_val_acc']:.4f} | Params: {res['search_params']}")
    print("="*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparameter Search Script")
    parser.add_argument('--type', type=str, default='random', choices=['grid', 'random'])
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--search_dir', type=str, default='')
    
    args = parser.parse_args()
    run_search(args)
