import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../..')


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class SantoriniNNet(nn.Module):
    def __init__(self, game, args):
        super(SantoriniNNet, self).__init__()
        _, self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.stem = nn.Sequential(
            nn.Conv2d(args.input_channels, args.num_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(args.num_channels),
            nn.ReLU(inplace=True),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(args.num_channels) for _ in range(args.num_residual_blocks)]
        )

        self.policy_conv = nn.Conv2d(args.num_channels, 2, kernel_size=1, stride=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.board_x * self.board_y, self.action_size)

        self.value_conv = nn.Conv2d(args.num_channels, 1, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.board_x * self.board_y, args.value_hidden_size)
        self.value_fc2 = nn.Linear(args.value_hidden_size, 1)

    def forward(self, s):
        s = self.stem(s)
        s = self.residual_blocks(s)

        pi = F.relu(self.policy_bn(self.policy_conv(s)))
        pi = pi.view(pi.size(0), -1)
        pi = self.policy_fc(pi)

        v = F.relu(self.value_bn(self.value_conv(s)))
        v = v.view(v.size(0), -1)
        v = F.dropout(F.relu(self.value_fc1(v)), p=self.args.dropout, training=self.training)
        v = self.value_fc2(v)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
