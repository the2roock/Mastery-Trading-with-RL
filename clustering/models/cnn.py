import torch
import torch.nn as nn

class Cnn(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()

        self.norm = nn.BatchNorm2d(d_model)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1) # 64 x 60 x 4
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=d_model, kernel_size=(3, 1), padding=(1,0)) # d_model x 60 x 4
        self.max_pool1 = nn.AvgPool2d(kernel_size=(2, 1)) # 64 x 30 x 4

        self.conv3 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(3, 1)) # d_model x 28 x 4
        self.conv4 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(3, 1)) # d_model x 26 x 4
        self.conv5 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(3, 1)) # d_model x 24 x 4

        self.conv6 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(1, 4)) # d_model x 24 x 1

        self.conv7 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(5,1)) # d_model x 20 x 1
        self.conv8 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(5,1)) # d_model x 16 x 1
        self.conv9 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(5,1)) # d_model x 12 x 1
        self.max_pool2 = nn.MaxPool2d(kernel_size=(4, 1)) # d_model x 3 x 1

        self.flatten = nn.Flatten()
        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

    def forward(self, x):
        x = x.reshape((*x.shape, 1)).permute(0, -1, 1, 2)

        l1 = torch.relu(self.conv1(x))
        l1 = torch.relu(self.conv2(l1))
        l1 = self.max_pool1(l1)

        l2 = torch.relu(self.conv3(l1))
        l2 = torch.relu(self.conv4(l2))
        l2 = torch.relu(self.conv5(l2))

        l3 = torch.relu(self.conv6(l2))

        l4 = torch.relu(self.conv7(l3))
        l4 = torch.relu(self.conv8(l4))
        l4 = torch.relu(self.conv9(l4))
        l4 = self.max_pool2(l4)

        l5 = l4.permute(0, 2, 3, 1).contiguous().view(l4.shape[0], l4.shape[2]*l4.shape[3], l4.shape[1])
        out, _ = self.attention(l5, l5, l5)
        return out
