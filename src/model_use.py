from models import models
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
import os.path as osp
import numpy as np
import utils as u

# Функция для загрузки данных
def load_data(data_path, noAgg=True):
    # Read edges, features and classes from csv files
    df_edges = pd.read_csv(osp.join(data_path, "elliptic_txs_edgelist.csv"))
    df_features = pd.read_csv(osp.join(data_path, "elliptic_txs_features.csv"), header=None)
    df_classes = pd.read_csv(osp.join(data_path, "elliptic_txs_classes.csv"))

    # Name colums basing on index
    colNames1 = {'0': 'txId', 1: "Time step"}
    colNames2 = {str(ii + 2): "Local_feature_" + str(ii + 1) for ii in range(94)}
    colNames3 = {str(ii + 96): "Aggregate_feature_" + str(ii + 1) for ii in range(72)}

    colNames = dict(colNames1, **colNames2, **colNames3)
    colNames = {int(jj): item_kk for jj, item_kk in colNames.items()}

    # Rename feature columns
    df_features = df_features.rename(columns=colNames)
    if noAgg:
        df_features = df_features.drop(df_features.iloc[:, 96:], axis=1)

    # Map unknown class to '3'
    df_classes.loc[df_classes['class'] == 'unknown', 'class'] = '3'

    print('df_classes')
    print(df_classes)
    print('df_features')
    print(df_features)

    # Merge classes and features in one Dataframe
    df_class_feature = pd.merge(df_classes, df_features)

    # df_class_feature = df_class_feature.fillna(0)

    print('df_class_feature')
    print(df_class_feature)

    # Exclude records with unknown class transaction

    df_class_feature = df_class_feature[df_class_feature["class"] != '3']

    # Build Dataframe with head and tail of transactions (edges)
    known_txs = df_class_feature["txId"].values
    df_edges = df_edges[(df_edges["txId1"].isin(known_txs)) & (df_edges["txId2"].isin(known_txs))]

    # Build indices for features and edge types
    features_idx = {name: idx for idx, name in enumerate(sorted(df_class_feature["txId"].unique()))}
    class_idx = {name: idx for idx, name in enumerate(sorted(df_class_feature["class"].unique()))}

    # Apply index encoding to features
    df_class_feature["txId"] = df_class_feature["txId"].apply(lambda name: features_idx[name])
    df_class_feature["class"] = df_class_feature["class"].apply(lambda name: class_idx[name])

    # Apply index encoding to edges
    df_edges["txId1"] = df_edges["txId1"].apply(lambda name: features_idx[name])
    df_edges["txId2"] = df_edges["txId2"].apply(lambda name: features_idx[name])

    return df_class_feature, df_edges

def data_to_pyg(df_class_feature, df_edges):
    # Преобразуем все признаки в числовые значения и заменяем NaN на 0
    features = df_class_feature.iloc[:, 3:].apply(pd.to_numeric, errors='coerce').fillna(0).values
    edge_index = torch.tensor([df_edges["txId1"].values, df_edges["txId2"].values], dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(df_class_feature["class"].values, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    data = RandomNodeSplit(num_test=1.0)(data)

    return data

# Загрузка обученной модели
def load_trained_model(model, model_path, args):
    for name, param in model.state_dict().items():
        print(f"{name}: {param.shape}")
    checkpoint = torch.load(model_path, map_location=args.device)
    model.load_state_dict(checkpoint)
    model.eval()  # Переводим модель в режим тестирования
    return model

# Основной код для загрузки данных, тестирования на выбранных узлах
@torch.no_grad()
def test_selected_nodes(data_path, model_path, selected_txids):
    args = u.get_config()

    # Загружаем данные
    df_class_feature, df_edges = load_data(data_path)
    data = data_to_pyg(df_class_feature, df_edges)

    print("Original data loaded:")
    print(data)

    # Фильтруем данные по выбранным txId
    selected_nodes = df_class_feature[df_class_feature["txId"].isin(selected_txids)]["txId"].values
    selected_nodes_tensor = torch.tensor(selected_nodes, dtype=torch.long)
    print(f"Selected nodes: {selected_nodes}")

    # Фильтруем x и y только для выбранных узлов
    selected_x = data.x[selected_nodes]
    selected_y = data.y[selected_nodes]

    # Фильтруем edges для выбранных узлов
    selected_edge_index = data.edge_index[:,
        torch.isin(data.edge_index[0], selected_nodes_tensor) & torch.isin(data.edge_index[1], selected_nodes_tensor)
    ]
    print(f"Filtered edge_index: {selected_edge_index.shape}")

    # Загружаем модель
    model = models.ChebyshevConvolution(args, [1, 2, 3, 4], 94, 64).to(args.device)
    model = load_trained_model(model, model_path, args)

    # Прогоняем модель только для выбранных узлов
    out, _ = model((selected_x, selected_edge_index))

    # Применяем softmax, чтобы получить вероятности для всех классов
    probabilities = F.softmax(out, dim=1)
    print("\nPredicted probabilities for selected nodes:")
    print(probabilities)

    # Получаем метки с максимальной вероятностью
    predicted_classes = probabilities.argmax(dim=1)
    print("\nPredictions for selected nodes:")
    for node, predicted_class in zip(selected_nodes, predicted_classes):
        print(f"Node {node} is predicted to belong to class {predicted_class.item()}")

    # Рассчитываем точность (accuracy) для тестового набора
    acc = u.accuracy(predicted_classes, selected_y)
    print(f"Accuracy on selected nodes: {acc.item()}")

    # Выводим реальную метку для выбранных узлов
    print("\nTrue labels for selected nodes:")
    for node, true_label in zip(selected_nodes, selected_y):
        print(f"Node {node} has true class {true_label.item()}")

# Параметры
data_path = '.\data\elliptic\dataset\elliptic_plus'
model_path = "aml_bitcoin.pth"  # Путь к модели
selected_nodes = [0, 10, 50, 100, 200, 40933]  # Узлы, которые хотим протестировать

# Запуск тестирования
test_selected_nodes(data_path, model_path, selected_nodes)