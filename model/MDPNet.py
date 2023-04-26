# coding = utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import r2_score
import copy


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

def evaluate(data_loader, net, criterion):
    net.eval()
    eval_loss = 0
    eval_r2 = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            " Prepare data "
            eval_x, eval_y = data
            if torch.cuda.is_available():
                eval_x = eval_x.cuda(0)
                eval_y = eval_y.cuda(0)
            " Forward "
            out = net(eval_x)
            loss = criterion(out, eval_y)
            eval_loss += loss.data.item()

            true = eval_y.cpu().numpy()
            pred = out.cpu().detach().numpy()
            eval_r2 += r2_score(true, pred)

        eval_loss = eval_loss / len(data_loader)
        eval_r2 = eval_r2 / len(data_loader)
        return eval_loss, eval_r2

def train(train_loader, val_loader, test_loader, num_epochs, criterion, optimizer):
    model = MDP_Net(768, 8, 512, 512, 256, 256, 1)
    if torch.cuda.is_available():
        model = model.cuda()

    val_r2_max = -1000
    train_loss_list = []  # 用于画图
    val_loss_list = []
    train_r2_list = []
    val_r2_list = []

    print("Start Training...")
    add = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        for i, data in enumerate(train_loader, 0):
            " Prepare data "
            train_x, train_y = data
            if torch.cuda.is_available():
                train_x = train_x.cuda(0)
                train_y = train_y.cuda(0)
            " Forward "
            out = model(train_x)
            loss = criterion(out, train_y)
            train_loss = loss.data.item()
            if i == 10:
                print("-----------------------------")
                print(epoch, i, train_loss)
            " Backward "
            optimizer.zero_grad()  # 梯度置0
            loss.backward()
            " Update "
            optimizer.step()

        if epoch % 1 == 0:
            val_loss, val_r2 = evaluate(val_loader, model, criterion)  # evaluation of validation set
            train_loss1, train_r2 = evaluate(train_loader, model, criterion)  # evaluation of training set
            val_loss_list.append(val_loss)
            train_loss_list.append(train_loss)
            val_r2_list.append(val_r2)
            train_r2_list.append(train_r2)

            if val_r2 > val_r2_max:
                val_r2_max = val_r2
                model_best = copy.deepcopy(model)
                epoch_best = epoch
            print(
                f'epoch: {epoch}, Train Loss:{train_loss:.6f}, Train R2:{train_r2:.6f}, Val Loss:{val_loss:.6f}, Val R2:{val_r2:.6f}')

        if len(train_loss_list) >= 2:
            add = add+1 if train_loss_list[-1] > train_loss_list[-2] else 0
            if add >= 5:
                print("Exit Training...")
                print(epoch_best, val_r2_max)
                test_loss, test_r2 = evaluate(test_loader, model, criterion)  # evaluation of testing set
                print(f'Results----Test Loss:{test_loss:.6f}, Test R2:{test_r2:.6f}')
                break

    return model_best






