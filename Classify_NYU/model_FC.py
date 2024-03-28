import torch.nn as nn
from torch.nn import functional as F
import torch
from torchsummary import summary
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            # class torch.nn.Conv1d(in_channels, out_channels, kernel_size,
            # stride=1, padding=0, dilation=1, groups=1, bias=True)
            nn.Conv1d(in_channels=11325, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 64, 1),
            nn.BatchNorm1d(64),
        )
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(11325, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)  # 生成模型
    input_size = torch.randn(4, 11325, 1)  # 随机初始化一个tensor做测试数据
    summary(model, input_size=(11325, 1), batch_size=4)
    pass