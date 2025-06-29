import torch
import torch.nn as nn
import torch.nn.functional as F

from config import gamma, device, layer_size


class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(QNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden = layer_size

        self.fc1 = nn.Linear(num_inputs, self.num_hidden)
        self.fc2 = nn.Linear(self.num_hidden, self.num_hidden)
        self.fc3 = nn.Linear(self.num_hidden, self.num_outputs)
        self.random_init()

    def random_init(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)  # Imposta il seed per PyTorch
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)  # Imposta il seed per CUDA

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def zero_init(self):
        """
        Inizializza tutti i pesi e i bias a zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)  # Inizializza i pesi a zero
                nn.init.constant_(m.bias, 0)  # Inizializza i bias a zero

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNQNet(nn.Module):
    def __init__(self, in_channels=4, n_actions=6):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(CNNQNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)

    # def optimize_model(online_net, target_net, optimizer, batch):
    #     states = torch.stack(batch.state)
    #     next_states = torch.stack(batch.next_state)
    #     actions = torch.Tensor(batch.action).to(device)
    #     rewards = torch.Tensor(batch.reward).to(device)
    #     masks = torch.Tensor(batch.mask).to(device)

    #     pred = online_net(states).squeeze(1)

    #     next_pred = target_net(next_states).squeeze(1)
    #     # print("pre",pred)
    #     pred = torch.sum(pred.mul(actions), dim=1)

    #     target = rewards + masks * gamma * next_pred.max(1)[0]

    #     criterion = nn.MSELoss()
    #     loss = criterion(pred,target)

    #     optimizer.zero_grad()
    #     loss.backward()

    #     torch.nn.utils.clip_grad_value_(online_net.parameters(), 10)
    #     optimizer.step()

    #     return loss

    # def get_action(self, input):
    #     qvalue = self.forward(input)
    #     _, action = torch.max(qvalue, 1)
    #     return action.item()
