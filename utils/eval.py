import torch
from tqdm import tqdm

def evaluate(model, dataloader, criterion, device, return_preds=False):
    """
    在测试集/验证集上评估模型的逻辑。
    :param return_preds: 是否返回所有的预测结果和真实标签，用于混淆矩阵或错例分析。
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        with tqdm(dataloader, desc="Evaluating", leave=False) as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 如果需要返回预测值，则记录当前的预测和标签
                if return_preds:
                    all_preds.append(predicted.cpu())
                    all_labels.append(labels.cpu())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    if return_preds:
        # 将 list 拼接为完整的 tensor
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    return epoch_loss, epoch_acc
