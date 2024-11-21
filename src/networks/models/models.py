from torch_geometric.nn import ChebConv
from torch.nn import Module
import torch.nn.functional as F


class ChebyshevConvolution(Module):
    def __init__(self, args, kernel, num_features, hidden_units):
        super().__init__()
        self.conv1 = ChebConv(num_features, hidden_units, kernel[0])
        self.conv2 = ChebConv(hidden_units, hidden_units, kernel[1])  # Новый скрытый слой
        self.conv3 = ChebConv(hidden_units, hidden_units, kernel[2])  # Новый скрытый слой
        self.conv4 = ChebConv(hidden_units, args['num_classes'], kernel[3])  # Выходной слой

    def forward(self, data):
        x, edge_index = data
        x = F.relu6(self.conv1(x, edge_index))  # Первый слой
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu6(self.conv2(x, edge_index))  # Второй слой
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu6(self.conv3(x, edge_index))  # Второй слой
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv4(x, edge_index)  # Третий (выходной) слой
        return x, edge_index