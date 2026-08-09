import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_unit import BehaviorEncoder, ResidualBlock, PolicyNetwork, ValueNetwork, DiscreteValueNetwork


class AlphaZeroNetwork(nn.Module):
    def __init__(self,
                 game_name,
                 num_input_channels,
                 input_channel_height,
                 input_channel_width,
                 num_hidden_channels,
                 hidden_channel_height,
                 hidden_channel_width,
                 num_blocks,
                 action_size,
                 num_value_hidden_channels,
                 discrete_value_size,
                 num_players=2,
                 use_rank_head=False,
                 use_behavior_conditioning=False,
                 behavior_history_length=16,
                 behavior_embedding_dim=32,
                 behavior_history_dropout=0.0):
        super(AlphaZeroNetwork, self).__init__()
        self.game_name = game_name
        self.num_input_channels = num_input_channels
        self.input_channel_height = input_channel_height
        self.input_channel_width = input_channel_width
        self.num_hidden_channels = num_hidden_channels
        self.hidden_channel_height = hidden_channel_height
        self.hidden_channel_width = hidden_channel_width
        self.num_blocks = num_blocks
        self.action_size = action_size
        self.num_value_hidden_channels = num_value_hidden_channels
        self.discrete_value_size = discrete_value_size
        self.num_players = num_players
        self.use_rank_head = use_rank_head and num_players > 2 and discrete_value_size == 1
        self.use_behavior_conditioning = use_behavior_conditioning and num_players > 2
        self.behavior_history_length = max(1, behavior_history_length)
        self.behavior_embedding_dim = max(1, behavior_embedding_dim)
        self.behavior_history_dropout = min(1.0, max(0.0, behavior_history_dropout))

        self.conv = nn.Conv2d(num_input_channels, num_hidden_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_hidden_channels)
        self.residual_blocks = nn.ModuleList([ResidualBlock(num_hidden_channels) for _ in range(num_blocks)])
        self.behavior_encoder = BehaviorEncoder(action_size, self.behavior_history_length, self.behavior_embedding_dim)
        self.behavior_relative_player_embedding = nn.Embedding(num_players, self.behavior_embedding_dim)
        self.behavior_query = nn.Linear(self.behavior_embedding_dim, self.behavior_embedding_dim)
        self.behavior_key = nn.Linear(self.behavior_embedding_dim, self.behavior_embedding_dim)
        self.behavior_dropout = nn.Dropout(self.behavior_history_dropout)
        self.behavior_film = nn.Linear(2 * self.behavior_embedding_dim, 2 * num_hidden_channels)
        nn.init.zeros_(self.behavior_film.weight)
        nn.init.zeros_(self.behavior_film.bias)
        self.policy = PolicyNetwork(num_hidden_channels, hidden_channel_height, hidden_channel_width, action_size,
                                    fully_convolutional=(game_name in ("blokus", "blokus15")))
        # Keep the attribute available for TorchScript even when the optional
        # rank head is disabled. Identity has no parameters and is never used
        # by forward() unless use_rank_head is true.
        self.rank = nn.Identity()
        if self.discrete_value_size == 1:
            value_output_size = num_players if num_players > 2 else 1
            self.value = ValueNetwork(num_hidden_channels, hidden_channel_height, hidden_channel_width, num_value_hidden_channels, value_output_size)
            if self.use_rank_head:
                self.rank = ValueNetwork(num_hidden_channels, hidden_channel_height, hidden_channel_width, num_value_hidden_channels, value_output_size)
        else:
            self.value = DiscreteValueNetwork(num_hidden_channels, hidden_channel_height, hidden_channel_width, num_value_hidden_channels, discrete_value_size)

    @torch.jit.export
    def get_type_name(self):
        return "alphazero"

    @torch.jit.export
    def get_game_name(self):
        return self.game_name

    @torch.jit.export
    def get_num_input_channels(self):
        return self.num_input_channels

    @torch.jit.export
    def get_input_channel_height(self):
        return self.input_channel_height

    @torch.jit.export
    def get_input_channel_width(self):
        return self.input_channel_width

    @torch.jit.export
    def get_num_hidden_channels(self):
        return self.num_hidden_channels

    @torch.jit.export
    def get_hidden_channel_height(self):
        return self.hidden_channel_height

    @torch.jit.export
    def get_hidden_channel_width(self):
        return self.hidden_channel_width

    @torch.jit.export
    def get_num_blocks(self):
        return self.num_blocks

    @torch.jit.export
    def get_action_size(self):
        return self.action_size

    @torch.jit.export
    def get_num_value_hidden_channels(self):
        return self.num_value_hidden_channels

    @torch.jit.export
    def get_discrete_value_size(self):
        return self.discrete_value_size

    @torch.jit.export
    def get_num_players(self):
        return self.num_players

    @torch.jit.export
    def get_use_rank_head(self):
        return self.use_rank_head

    @torch.jit.export
    def get_behavior_history_length(self):
        return self.behavior_history_length if self.use_behavior_conditioning else 0

    @torch.jit.export
    def get_behavior_embedding_dim(self):
        return self.behavior_embedding_dim if self.use_behavior_conditioning else 0

    def forward(self, state, behavior_history, to_play):
        x = self.conv(state)
        x = self.bn(x)
        x = F.relu(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x)

        if self.use_behavior_conditioning:
            player_tokens = self.behavior_encoder(behavior_history)
            batch_index = torch.arange(state.shape[0], device=state.device)
            player_index = torch.arange(self.num_players, device=state.device).view(1, -1)
            relative_player = torch.remainder(player_index - to_play.view(-1, 1), self.num_players)
            player_tokens = player_tokens + self.behavior_relative_player_embedding(relative_player)
            current_token = player_tokens[batch_index, to_play]
            attention = (self.behavior_query(current_token).unsqueeze(1) * self.behavior_key(player_tokens)).sum(dim=2)
            current_mask = torch.nn.functional.one_hot(to_play, num_classes=self.num_players).to(torch.bool)
            attention = torch.softmax(attention.masked_fill(current_mask, -10000.0), dim=1)
            opponent_token = (attention.unsqueeze(2) * player_tokens).sum(dim=1)
            behavior_context = self.behavior_dropout(torch.cat((current_token, opponent_token), dim=1))
            gamma, beta = self.behavior_film(behavior_context).chunk(2, dim=1)
            x = x * (1.0 + torch.tanh(gamma).view(-1, self.num_hidden_channels, 1, 1)) + beta.view(-1, self.num_hidden_channels, 1, 1)

        # policy
        policy_logit = self.policy(x)
        policy = torch.softmax(policy_logit, dim=1)

        # value
        if self.discrete_value_size == 1:
            value = self.value(x)
            output = {"policy_logit": policy_logit,
                      "policy": policy,
                      "value": value}
            if self.use_rank_head:
                output["rank"] = self.rank(x)
            return output
        else:
            value_logit = self.value(x)
            value = torch.softmax(value_logit, dim=1)
            return {"policy_logit": policy_logit,
                    "policy": policy,
                    "value_logit": value_logit,
                    "value": value}
