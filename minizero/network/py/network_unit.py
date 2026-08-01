import math
import torch.nn as nn
import torch.nn.functional as F
import torch


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        input = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(input + x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_channels, channel_height, channel_width, action_size, fully_convolutional=False):
        super(PolicyNetwork, self).__init__()
        self.channel_height = channel_height
        self.channel_width = channel_width
        self.num_output_channels = math.ceil(action_size / (channel_height * channel_width))
        self.conv = nn.Conv2d(num_channels, self.num_output_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(self.num_output_channels)
        self.fully_convolutional = fully_convolutional
        if fully_convolutional:
            assert self.num_output_channels * channel_height * channel_width == action_size
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(self.num_output_channels * channel_height * channel_width, action_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(-1, self.num_output_channels * self.channel_height * self.channel_width)
        x = self.fc(x)
        return x


class BehaviorEncoder(nn.Module):
    def __init__(self, action_size, history_length, embedding_dim):
        super(BehaviorEncoder, self).__init__()
        self.action_size = action_size
        self.history_length = history_length
        self.embedding_dim = embedding_dim
        self.action_embedding = nn.Embedding(action_size + 1, embedding_dim, padding_idx=action_size)
        self.position_embedding = nn.Embedding(history_length, embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.output = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, history):
        batch_size = history.shape[0]
        num_players = history.shape[1]
        mask = history.ne(self.action_size)
        positions = torch.arange(self.history_length, device=history.device).view(1, 1, -1)
        tokens = self.action_embedding(history) + self.position_embedding(positions)
        flat_tokens = tokens.view(batch_size * num_players, self.history_length, self.embedding_dim)
        flat_mask = mask.view(batch_size * num_players, self.history_length)

        query = self.query(flat_tokens)
        key = self.key(flat_tokens)
        value = self.value(flat_tokens)
        attention = torch.matmul(query, key.transpose(1, 2)) / float(self.embedding_dim) ** 0.5
        attention = attention.masked_fill(~flat_mask.unsqueeze(1), -10000.0)
        attention = torch.softmax(attention, dim=2)
        contextual = self.output(torch.matmul(attention, value))
        contextual = contextual * flat_mask.unsqueeze(2).float()
        denominator = flat_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        pooled = contextual.sum(dim=1) / denominator
        return pooled.view(batch_size, num_players, self.embedding_dim)


class ValueNetwork(nn.Module):
    def __init__(self, num_channels, channel_height, channel_width, num_value_hidden_channels, output_size=1):
        super(ValueNetwork, self).__init__()
        self.channel_height = channel_height
        self.channel_width = channel_width
        self.conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(channel_height * channel_width, num_value_hidden_channels)
        self.fc2 = nn.Linear(num_value_hidden_channels, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(-1, self.channel_height * self.channel_width)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x


class DiscreteValueNetwork(nn.Module):
    def __init__(self, num_channels, channel_height, channel_width, num_value_hidden_channels, value_size):
        super(DiscreteValueNetwork, self).__init__()
        self.channel_height = channel_height
        self.channel_width = channel_width
        self.hidden_channels = math.ceil(value_size / (channel_height * channel_width))
        self.conv = nn.Conv2d(num_channels, self.hidden_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(self.hidden_channels)
        self.fc1 = nn.Linear(channel_height * channel_width * self.hidden_channels, num_value_hidden_channels)
        self.fc2 = nn.Linear(num_value_hidden_channels, value_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(-1, self.channel_height * self.channel_width * self.hidden_channels)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
