from torch import nn

class GenerateScoreNet(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(GenerateScoreNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)  
        self.fc2 = nn.Linear(512, 256)         
        self.fc3 = nn.Linear(256, output_size) 
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, combined_features):
        # 使用ReLU激活函数处理前两层网络的输出
        x = self.relu(self.fc1(combined_features))
        x = self.relu(self.fc2(x))
        
        # 将输出通过 Sigmoid 激活函数转换为 [0, 10] 范围内的分数
        score = self.sigmoid(self.fc3(x)) * 10
        
        return score