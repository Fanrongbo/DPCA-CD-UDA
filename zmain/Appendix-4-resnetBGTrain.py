import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torchvision.models.resnet import ResNet50_Weights

SEED=102
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# 1. 数据加载
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root='/home/yjh/DatasetF/CD_DA_building/NWPU-RESISC45/', transform=data_transform)
# 打印文件夹名称及其对应的label
# 获取文件夹名称及其对应的label的字典
folder_to_label_dict = dataset.class_to_idx
print(folder_to_label_dict)
# for class_name, label in dataset.class_to_idx.items():
#     print(f"Folder Name: {class_name}, Label: {label}")
# 使用random_split按9:1的比例划分数据集
train_len = int(0.9 * len(dataset))
val_len = len(dataset) - train_len
train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# 2. 定义模型
# model = models.resnet50(pretrained=True)  # 使用预训练的ResNet50
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

model.fc = nn.Linear(model.fc.in_features, 45)  # 修改最后的全连接层以匹配45个类别
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # 5. 验证模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")
torch.save(model.state_dict(), 'resnetBG.pth')
