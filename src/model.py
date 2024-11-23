import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self, dropout_rate = 0):
        super(MNISTModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.ant1 = nn.Conv2d(32,8,kernel_size=1)

        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.conv5 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.dropout5 = nn.Dropout(dropout_rate)

        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.ant2 = nn.Conv2d(32,8,kernel_size=1)

        self.conv6 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(16)         
        self.dropout6 = nn.Dropout(dropout_rate)
        self.conv7 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.avgpool = nn.AvgPool2d(2, 2)

        self.fc = nn.Linear(32 * 3 * 3, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout1(self.bn1(self.relu(self.conv1(x))))
        x = self.dropout2(self.bn2(self.relu(self.conv2(x))))
        x = self.dropout3(self.bn3(self.relu(self.conv3(x))))
    
        x = self.maxpool1(self.relu(x))
        x = self.ant1(x)

        x = self.dropout4(self.bn4(self.relu(self.conv4(x))))
        x = self.dropout5(self.bn5(self.relu(self.conv5(x))))

        x = self.maxpool2(self.relu(x))
        x = self.ant2(x)

        x = self.dropout6(self.bn6(self.relu(self.conv6(x))))
        x = self.relu(self.conv7(x))
        x = self.avgpool(x)
      
        x = x.view(-1, 32 * 3 * 3)    
        x = self.fc(x)
        return torch.log_softmax(x, dim=1) 
    
    def to_device(self):
        return self.to('cpu')

    
if __name__ == "__main__":
    model = MNISTModel()
    print(summary(model, (1, 28, 28)))
