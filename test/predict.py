import torch
import os
from PIL import Image
from torchvision import transforms
from models import SimpleDLA

# 设备配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# CIFAR-10 的10个类别
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 加载模型
net = SimpleDLA()
net = net.to(device)
net = torch.nn.DataParallel(net)
checkpoint = torch.load('./checkpoint/best_ckpt.pth', map_location=device)
net.load_state_dict(checkpoint['net'])
net.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# 图片路径
img_dir = './test_images'
img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# 正确预测计数
correct = 0
total = 0

for img_file in img_files:
    img_path = os.path.join(img_dir, img_file)
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f'{img_file} 打开失败：{e}')
        continue

    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = net(input_tensor)
        pred = output.argmax(dim=1).item()

    print(f'{img_file} -> 预测类别：{classes[pred]}')

    # 检查文件名中是否包含类别关键词
    for i, cls in enumerate(classes):
        if cls in img_file.lower():
            total += 1
            if pred == i:
                correct += 1
            break

# 输出准确率
if total > 0:
    acc = 100 * correct / total
    print(f'\n识别有效图片共 {total} 张，预测正确 {correct} 张，准确率为：{acc:.2f}%')
else:
    print('\n未能从文件名中识别出任何有效类别，无法计算准确率')
