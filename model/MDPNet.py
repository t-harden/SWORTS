# coding = utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn


class MDP_Net(nn.Module):
    def __init__(self, embedding_size, PathAttention_factor, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, out_dim):
        super(MDP_Net, self).__init__()
        " The embedding of low-level and high-level workflows "
        self.PathAttention_factor = PathAttention_factor

        self.attention_W1 = nn.Parameter(torch.Tensor(
            embedding_size, self.PathAttention_factor))
        self.attention_b1 = nn.Parameter(torch.Tensor(self.PathAttention_factor))
        self.projection_h1 = nn.Parameter(
            torch.Tensor(self.PathAttention_factor, 1))
        for tensor in [self.attention_W1, self.projection_h1]:
            nn.init.xavier_normal_(tensor, )
        for tensor in [self.attention_b1]:
            nn.init.zeros_(tensor, )

        self.attention_W2 = nn.Parameter(torch.Tensor(
            embedding_size, self.PathAttention_factor))
        self.attention_b2 = nn.Parameter(torch.Tensor(self.PathAttention_factor))
        self.projection_h2 = nn.Parameter(
            torch.Tensor(self.PathAttention_factor, 1))
        for tensor in [self.attention_W2, self.projection_h2]:
            nn.init.xavier_normal_(tensor, )
        for tensor in [self.attention_b2]:
            nn.init.zeros_(tensor, )
        " Matching Degree Prediction "
        in_dim = embedding_size * 3
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), nn.BatchNorm1d(n_hidden_4))
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, out_dim))


    def forward(self, inputs):
        " The embedding of low-level and high-level workflows "
        path_num = (inputs.shape[1] - 2)/2

        attention_input1 = inputs[:, 2:path_num + 2, :]
        attention_temp1 = F.relu(torch.tensordot(
            attention_input1, self.attention_W1, dims=([-1], [0])) + self.attention_b1)
        self.normalized_att_score1 = F.softmax(torch.tensordot(
            attention_temp1, self.projection_h1, dims=([-1], [0])), dim=1)
        attention_output1 = torch.sum(
            self.normalized_att_score1 * attention_input1, dim=1)

        attention_input2 = inputs[:, path_num + 2:, :]
        attention_temp2 = F.relu(torch.tensordot(
            attention_input2, self.attention_W2, dims=([-1], [0])) + self.attention_b2)
        self.normalized_att_score2 = F.softmax(torch.tensordot(
            attention_temp2, self.projection_h2, dims=([-1], [0])), dim=1)
        attention_output2 = torch.sum(
            self.normalized_att_score2 * attention_input2, dim=1)

        workflow_embed = attention_output1 * attention_output2

        " Matching Degree Prediction "
        demand_embed = inputs[:, 0, :]
        external_embed = inputs[:, 1, :]
        pred_input = torch.cat((demand_embed, external_embed, workflow_embed), dim=1)

        hidden_1_out = F.relu(self.layer1(pred_input))
        hidden_2_out = F.relu(self.layer2(hidden_1_out))
        hidden_3_out = F.relu(self.layer3(hidden_2_out))
        hidden_4_out = F.relu(self.layer4(hidden_3_out))

        out = torch.sigmoid(self.layer5(hidden_4_out))

        return out






