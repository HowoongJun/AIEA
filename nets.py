import torch

class CAieaNets(torch.nn.Module):
    def __init__(self):
        super(CActiveSceneExplorer, self).__init__()
        self.relu = torch.nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=7)

        self.conv1_1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)       
        self.conv3_2 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.max_pool(x)
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.max_pool(x)
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.max_pool(x)
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        features = self.max_pool(x)
    
        return features