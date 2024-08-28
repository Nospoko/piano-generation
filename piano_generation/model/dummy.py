import torch
from torch import nn as nn


# Dummy model for demonstration purposes
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 'VELOCITY_31', 'NOTE_ON_56', '7T', '6T', '3T', 'VELOCITY_31', 'NOTE_OFF_56'
        self.note = [334, 197, 341, 340, 337, 334, 198]
        self.it = 0

    def generate_new_tokens(self, idx, max_new_tokens, temperature):
        out = torch.tensor([self.note[self.it : self.it + max_new_tokens]])
        self.it += max_new_tokens
        if self.it > 6:
            self.it = 0
        return out


# Dummy model for debugging
class RepeatingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_id = 25

    def generate_new_tokens(self, idx: torch.tensor, max_new_tokens, temperature):
        task_token_id = list(idx[0]).index(self.token_id)
        out = idx[:, idx.shape[1] - task_token_id : idx.shape[1] - task_token_id + max_new_tokens]
        return out
