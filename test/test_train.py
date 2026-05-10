import os
import sys
import json
import unittest
import tempfile
import argparse
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入你的训练入口函数
from train import run_training

def mock_get_train_val_dataloaders(batch_size=4, image_size=224, num_workers=0):
    """
    创建一个微型的伪造数据集用于测试。
    包含 8 张随机生成的图片和固定的标签。
    """
    print("\n[Mock] 正在生成微型测试数据集...")
    # 生成随机图片张量 (N, C, H, W)
    mock_images = torch.randn(8, 3, image_size, image_size)
    # 随机分配 0~36 之间的标签
    mock_labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    
    # 包装成 Dataset 和 DataLoader
    dataset = TensorDataset(mock_images, mock_labels)
    # 为了更容易过拟合，我们不 shuffle
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 训练集和验证集使用同一个 loader
    return loader, loader

class TestTrainingPipeline(unittest.TestCase):
    
    # 拦截 train.py 中调用的 get_train_val_dataloaders 函数
    @patch('train.get_train_val_dataloaders', side_effect=mock_get_train_val_dataloaders)
    def test_overfit_small_dataset(self, mock_dataloader_func):
        """
        测试目标：模型能否在极小的数据集上过拟合。
        判断标准：10个 Epoch 后，训练集的 Loss 必须显著下降，Acc 应该接近 100%。
        """
        # 使用临时文件夹存放测试输出，测试结束后自动清理
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # 1. 构造模拟的命令行参数 (Namespace)
            # 为了快速过拟合，我们设置较大的学习率，并禁用预训练权重（从头学习随机数据）
            args = argparse.Namespace(
                save_dir=temp_dir,
                save_history=True,
                resume='',
                model_name='resnet18',
                pretrained=False,        # 测试时不下载预训练权重，节省时间
                epochs=10,               # 跑 10 个 Epoch 足以让 8 张图过拟合
                batch_size=4,
                num_workers=0,
                optimizer='AdamW',
                lr_backbone=1e-3,        # 刻意调大学习率以加快过拟合
                lr_head=1e-3,
                weight_decay=0.0,        # 关掉正则化，允许模型尽过拟合数据
                lr_scheduler='None'      # 不调整学习率
            )
            
            print("\n=== 开始执行 Overfit Sanity Check ===")
            # 2. 运行主训练流程
            run_training(args)
            
            # 3. 验证 mock 是否被成功调用
            self.assertTrue(mock_dataloader_func.called, "Mock Dataloader 未被调用！")
            
            # 4. 读取生成的历史记录
            history_file = os.path.join(temp_dir, 'history.json')
            self.assertTrue(os.path.exists(history_file), "未能生成 history.json")
            
            with open(history_file, 'r') as f:
                history_data = json.load(f)

            epoch_history = history_data['epoch_history']

            self.assertEqual(len(epoch_history), args.epochs, "保存的 Epoch 数量与配置不符")
            
            # 5. 断言指标：Loss 必须下降，Acc 必须上升
            first_epoch = epoch_history[0]
            last_epoch = epoch_history[-1]
            
            print("\n=== Sanity Check 结果 ===")
            print(f"初始 Train Loss: {first_epoch['train_loss']:.4f} | Acc: {first_epoch['train_acc']:.4f}")
            print(f"最终 Train Loss: {last_epoch['train_loss']:.4f} | Acc: {last_epoch['train_acc']:.4f}")
            
            # 断言 1: 最终的 Loss 应该小于初始 Loss
            self.assertLess(last_epoch['train_loss'], first_epoch['train_loss'], 
                            "训练失败：模型经过 10 个 Epoch 后 Loss 没有下降！")
            
            # 断言 2: 最终准确率应该达到 100% (由于只有 8 个样本且关闭了权重衰减)
            self.assertGreaterEqual(last_epoch['train_acc'], 0.8, 
                                    "训练异常：在 8 张图片上准确率未能达到高水平，模型学习能力受限。")

if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
