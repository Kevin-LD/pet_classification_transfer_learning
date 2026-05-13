import wandb
import json
import os

def upload_replay(json_path, run_name, project_name="Model_Comparison_Replay"):
    """
    将本地 JSON 历史数据回放至 W&B
    """
    if not os.path.exists(json_path):
        print(f"跳过: 找不到文件 {json_path}")
        return

    # 1. 加载数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    epoch_history = data.get("epoch_history", [])
    iter_history = data.get("iter_history", [])

    # 2. 初始化 W&B Run
    # notes 可以写一些备注，tags 方便过滤
    run = wandb.init(
        project=project_name,
        name=run_name,
        config={
            "model_type": run_name,
            "source": "historical_json"
        }
    )

    print(f"正在上传 {run_name} 的数据...")

    # 3. 首先上传 Iteration 级别的 Loss
    # 我们使用一个计数器来模拟真实的 step
    global_step = 0
    for entry in iter_history:
        # log 只有 loss 的数据
        wandb.log({"iter/train_loss": entry["loss"]}, step=global_step)
        global_step += 1

    # 4. 然后上传 Epoch 级别的数据
    # 这里我们不需要传 step，或者继续沿用 global_step
    # 但关键是字典里要有 "epoch" 键，方便 W&B 后台对齐
    for entry in epoch_history:
        # 给所有的 key 加一个前缀，方便在 UI 中分类查看
        formatted_entry = {f"epoch/{k}": v for k, v in entry.items() if k != "epoch"}
        formatted_entry["epoch"] = entry["epoch"] # 保持 epoch 原名作为坐标轴
        
        # 建议：将 epoch 数据同步到最后的 global_step 
        # 或者直接 log，W&B 会默认记录当前时间戳
        wandb.log(formatted_entry)

    # 5. 结束 Run
    run.finish()
    print(f"{run_name} 上传完成！\n")

if __name__ == "__main__":
    # 在这里配置文件路径和对应的模型名称
    experiments = [
        # {"path": "runs/exp_20260511_132554_dataaug_0.12/history.json", "name": "Data Augmentation (Crop size lower bound: 0.12)"},
        # {"path": "runs/exp_20260511_133422_dataaug_0.08/history.json", "name": "Data Augmentation (Crop size lower bound: 0.08)"},
        {"path": "runs/exp_20260511_134100_dataaug_0.1/history.json", "name": "Data Augmentation (Crop size lower bound: 0.1)"},
    ]

    # 项目名称
    PROJECT = "Data Augmentation Comparison"

    for exp in experiments:
        upload_replay(exp["path"], exp["name"], PROJECT)
