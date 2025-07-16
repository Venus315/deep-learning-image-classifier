import torch
from core.model_custom_dla import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置运行设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型加载
model = (model(model))
model = torch.nn.DataParallel(model).to(device)
checkpoint = torch.load('./checkpoint/ckpt_epoch30.pth')
model.load_state_dict(checkpoint['net'])
model.eval()

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=100, shuffle=False)

# 初始化变量
true_labels, predicted_labels = [], []
batch_loss, batch_acc = [], []
correct = total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(preds.cpu().numpy())

        loss = torch.nn.functional.cross_entropy(outputs, labels)
        acc = 100.0 * correct / total

        batch_loss.append(loss.item())
        batch_acc.append(acc)

# 输出基本指标
final_acc = 100.0 * correct / total
print(f"✅ 测试集准确率: {final_acc:.2f}%")

prec = precision_score(true_labels, predicted_labels, average='weighted')
rec = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')
print(f"精度: {prec:.2f}")
print(f"召回率: {rec:.2f}")
print(f"F1 值: {f1:.2f}")

# 分类报告
print("分类报告：")
print(classification_report(true_labels, predicted_labels, target_names=testset.classes))

# 可视化：混淆矩阵
cm = confusion_matrix(true_labels, predicted_labels, normalize='true')
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=testset.classes, yticklabels=testset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 可视化：损失曲线
plt.figure(figsize=(10, 4))
plt.plot(batch_loss, label='Loss per Batch', color='red')
plt.title('Batch Loss Curve')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 可视化：准确率曲线
plt.figure(figsize=(10, 4))
plt.plot(batch_acc, label='Accuracy (%) per Batch', color='green')
plt.title('Batch Accuracy Curve')
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
