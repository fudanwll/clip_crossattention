import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from datetime import datetime
from CLIPCrossNet import CLIPCrossNet
from utils.Mydataset import MyDataset

model_path = './models/clip-vit-large-patch14'

os.makedirs('./result/trainlog', exist_ok=True)
os.makedirs('./result/pth', exist_ok=True)

# 参数设置
num_epochs = 512
batch_size = 16
learning_rate = 0.001

if torch.cuda.is_available():
    device_index = torch.cuda.current_device()
    print("当前设备索引:", device_index)

    # 获取当前 CUDA 设备的名称
    device_name = torch.cuda.get_device_name(device_index)
    print("当前设备名称:", device_name)
else:
    print("没有可用的 CUDA 设备")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#加载数据集
dataset = MyDataset(csv_file='data/labels.csv', img_dir='data/images/')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型
model = CLIPCrossNet(model_path = model_path, device = device)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# 训练过程
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for (image1, text1, image2, text2), targets in tqdm(dataloader, desc="Training", leave=False):
        image1, text1, image2, text2, targets = image1.to(device), text1.to(device), image2.to(device), text2.to(device), targets.to(device)

        # 清除优化器的梯度
        optimizer.zero_grad()

        # 前向传播
        scores = model(image1, text1, image2, text2)
        
        # 损失计算
        loss = criterion(scores, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

# 训练循环
for epoch in range(num_epochs):
    loss = train_epoch(model, dataloader, criterion, optimizer, device)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}')

    # 保存模型和日志
    torch.save(model.state_dict(), f'./result/pth/model_epoch_{epoch+1}.pth')
    with open('./result/trainlog/log.txt', 'a') as f:
        f.write(f'{datetime.now()} Epoch {epoch+1}, Loss: {loss:.4f}\n')

print("Training completed.")