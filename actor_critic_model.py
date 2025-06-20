"""
Generated code with interative prompts using
ChatGPT 40
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_space):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.policy = nn.Linear(128, action_space)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        policy_dist = F.softmax(self.policy(x), dim=-1)
        value = self.value(x)
        return policy_dist, value

