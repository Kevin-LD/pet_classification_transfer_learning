# Pet Classification with Transfer Learning
## 项目简介
本项目基于 PyTorch 实现 Oxford-IIIT Pet Dataset 宠物分类任务，包含迁移学习训练、超参数搜索、预训练消融实验、注意力模型性能对比与结果可视化分析。  
## 环境配置
实验环境：Ubuntu 22.04.5 LTS（WSL） + Python 3.12。  
使用以下命令安装项目依赖：  
```bash
pip install -r requirements.txt
```
## 数据准备
首次运行时会自动下载 Oxford-IIIT Pet Dataset，默认保存路径为 `./data`，无需手动准备数据集。  
## 运行方式
### 模型训练
```bash
python train.py --epochs 30 --batch_size 64 --lr_backbone 1e-5 --lr_head 1e-4 --weight_decay 1e-4
```
### 模型评估
```bash
python evaluate.py --model_path path/to/model.pth
```
更多可选参数可通过以下命令查看：  
```bash
python train.py -h
python evaluate.py -h
```
其余可视化脚本与超参数搜索脚本的使用方法同样可通过 `-h` 查看。  
## 模型权重
模型权重下载链接：https://drive.google.com/file/d/1s2ZIvFHIIrooQfqziZymulcBspTyGJWn/view?usp=sharing
## 相关仓库
- Task 2: https://github.com/Kevin-LD/visdrone_mot  
- Task 3: TODO  
- Report: https://github.com/Kevin-LD/cv_midterm_report  
