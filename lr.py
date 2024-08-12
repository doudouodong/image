import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

lr = 0.1

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

    def forward(self, x):
        pass


net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = StepLR(optimizer=optimizer, step_size=4, gamma=0.1)

print("初始化学习率: ", optimizer.defaults['lr'])

for epoch in range(1, 11):
    optimizer.zero_grad()
    optimizer.step()
    print("第%d个epoch的学习率: %f" % (epoch, optimizer.param_groups[0]['lr']))
    scheduler.step()
