import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import clip

class MyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, device='cuda'):
        """
        Args:
            csv_file (string): 图片描述的CSV文件路径。
            img_dir (string): 图片文件夹的路径。
            transform (callable, optional): 可选的转换操作，应用于图片。
            device (string): 计算设备，'cuda'或'cpu'。
        """
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.device = device

        # 预设的CLIP图像处理
        self.preprocess = clip.load("ViT-L/14", device=device, jit=False)[1]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")

        # 应用预处理
        if self.transform:
            image = self.transform(image)
        else:
            image = self.preprocess(image)

        text = self.img_labels.iloc[idx, 1]
        
        return image, text

# 使用示例
# transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # 根据模型输入调整大小
#         transforms.ToTensor(),          # 转换为torch tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     dataset = MyDataset(csv_file='dataset/labels.csv', img_dir='dataset/images/', transform=transform)
#     image, text = dataset[0]  # 加载第一个样本
#     print(image.shape, text)