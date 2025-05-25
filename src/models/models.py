from torch_geometric.nn import ChebConv  # Импорт слоя Chebyshev Convolution для работы с графами
from torch.nn import Module  # Базовый класс для всех моделей PyTorch
import torch.nn.functional as F  # Библиотека функций активации и других операций

class ChebyshevConvolution(Module):
    def __init__(self, args, kernel, num_features, hidden_units):
        """
        Инициализация модели сверточной нейронной сети на графах.

        Параметры:
        - args: словарь с гиперпараметрами, включает количество классов ('num_classes').
        - kernel: список размеров ядер свертки для каждого слоя.
        - num_features: количество входных признаков для каждого узла.
        - hidden_units: количество нейронов в скрытых слоях.
        """
        super().__init__()
        # Первый сверточный слой: преобразует входные признаки в скрытые признаки
        self.conv1 = ChebConv(num_features, hidden_units, kernel[0])
        # Второй сверточный слой: извлекает более сложные признаки
        self.conv2 = ChebConv(hidden_units, hidden_units, kernel[1])
        # Третий (выходной) слой: преобразует признаки в количество классов
        self.conv3 = ChebConv(hidden_units, args['num_classes'], kernel[2])

    def forward(self, data):
        """
        Прямой проход данных через модель.

        Параметры:
        - data: кортеж из признаков узлов (x) и структуры графа (edge_index).

        Возвращает:
        - x: выходные признаки узлов.
        - edge_index: структура графа, переданная без изменений.
        """
        x, edge_index = data  # Распаковываем входные данные: признаки и графовую структуру

        # Первый сверточный слой с функцией активации ReLU6
        x = F.relu6(self.conv1(x, edge_index))
        # Dropout для регуляризации (с вероятностью 50%)
        # x = F.dropout(x, p=0.1, training=self.training)

        # Второй сверточный слой с ReLU6
        x = F.relu6(self.conv2(x, edge_index))
        # Dropout для регуляризации
        # x = F.dropout(x, p=0.1, training=self.training)

        # Третий (выходной) слой, без функции активации
        x = self.conv3(x, edge_index)

        return x, edge_index  # Возвращаем выходные признаки и структуру графа
