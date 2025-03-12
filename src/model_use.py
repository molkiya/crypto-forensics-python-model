from models import models
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
import os.path as osp
import numpy as np
import utils as u
from train import test

u.seed_everything(42)

def data_to_pyg(df_class_feature, df_edges):
    # Преобразуем все признаки в числовые значения и заменяем NaN на 0
    features = df_class_feature.iloc[:, 3:].apply(pd.to_numeric, errors='coerce').fillna(0).values
    print('features')
    print(features)

    edge_index = torch.tensor([df_edges["txId1"].values, df_edges["txId2"].values], dtype=torch.long)
    print('edge_index')
    print(edge_index)

    x = torch.tensor(features, dtype=torch.float)
    print('data_to_pyg_x')
    print(x)

    y = torch.tensor(df_class_feature["class"].values, dtype=torch.long)
    print('data_to_pyg_y')
    print(y)

    data = Data(x=x, edge_index=edge_index, y=y)
    print('data_to_pyg_data')
    print(data)

    data = RandomNodeSplit(num_test=1.0)(data)
    print('data_to_pyg_after_random_node_split_data')
    print(data)

    return data

# Загрузка обученной модели
def load_trained_model(model, model_path, args):
    for name, param in model.state_dict().items():
        print(f"{name}: {param.shape}")
    checkpoint = torch.load(model_path, map_location=args.device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_data(data_path, selected_rows, noAgg=False):
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
    selected_df_class_feature = df_class_feature.iloc[selected_rows]
    df_class_feature.loc[df_class_feature["class"] == '3', "class"] = np.random.choice(['1', '2'], size=len(
        df_class_feature[df_class_feature["class"] == '3']))

    print('selected_df_class_feature')
    print(selected_df_class_feature)

    # Получаем список txId для выбранных строк
    selected_txs = set(map(int, selected_df_class_feature["txId"].values))

    print('selected_txs')
    print(selected_txs)

    # Находим транзакции, связанные с выбранными (смотрим в df_edges)
    connected_txs = set(df_edges[df_edges["txId1"].isin(selected_txs)]["txId2"]) | set(df_edges[df_edges["txId2"].isin(selected_txs)]["txId1"])

    print('connected_txs')
    print(connected_txs)

    # Объединяем выбранные транзакции и связанные с ними
    final_txs = selected_txs | connected_txs

    print('final_txs')
    print(final_txs)

    # Формируем новый df_class_feature, содержащий все нужные транзакции
    selected_df_class_feature = df_class_feature[df_class_feature["txId"].isin(final_txs)]

    print('selected_df_class_feature')
    print(selected_df_class_feature)

    # Фильтруем ребра: оставляем только те, где обе транзакции в final_txs
    final_df_edges = df_edges[(df_edges["txId1"].isin(final_txs)) & (df_edges["txId2"].isin(final_txs))]

    print('final_df_edges')
    print(final_df_edges)

    # Формируем индексы для транзакций и классов
    features_idx = {name: idx for idx, name in enumerate(sorted(selected_df_class_feature["txId"].unique()))}
    class_idx = {name: idx for idx, name in enumerate(sorted(selected_df_class_feature["class"].unique()))}

    print('features_idx')
    print(features_idx)

    print('class_idx')
    print(class_idx)

    # Кодируем идентификаторы транзакций и классы
    selected_df_class_feature["txId"] = selected_df_class_feature["txId"].apply(lambda name: features_idx[name])
    selected_df_class_feature["class"] = selected_df_class_feature["class"].apply(lambda name: class_idx[name])

    # Кодируем идентификаторы рёбер
    final_df_edges["txId1"] = final_df_edges["txId1"].apply(lambda name: features_idx[name])
    final_df_edges["txId2"] = final_df_edges["txId2"].apply(lambda name: features_idx[name])

    return selected_df_class_feature, final_df_edges

# ==================================================================================

def load_data_old(data_path, selected_rows, noAgg=False):
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

# Основной код для загрузки данных, тестирования на выбранных узлах
@torch.no_grad()
def test_selected_nodes(data_path, model_path, selected_rows):
    args = u.get_config()

    # Загружаем данные
    df_class_feature, df_edges = load_data(data_path, selected_rows, noAgg=True)
    print('df_class_feature_from_load_data')
    print(df_class_feature)
    print('df_edges_from_load_data')
    print(df_edges)
    data = data_to_pyg(df_class_feature, df_edges)

    print("\n=== Original Data ===")
    print("Graph Data Object:")
    print(data)  # Вывод PyG объекта данных
    # Загружаем модель
    model = models.ChebyshevConvolution(args, [1, 2, 3, 4], data.num_features, args.hidden_units_noAgg).to(args.device)
    model = load_trained_model(model, model_path, args)

    # Прогоняем модель только для выбранных узлов
    """Train a GNN model, return the trained model, and plot metrics."""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    else:
        print("No GPU available. Training will run on CPU.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    test(model, data, True)

# Параметры
data_path = '.\data\elliptic\dataset\elliptic_plus'
model_path = "aml_bitcoin.pth"  # Путь к модели
selected_nodes = [0, 10, 50, 100, 200, 40933, 44832, 44833, 44881]  # Узлы, которые хотим протестировать

# Запуск тестирования
test_selected_nodes(data_path, model_path, selected_nodes)