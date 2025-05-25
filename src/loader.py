import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit

import os.path as osp
import pandas as pd

def load_data(data_path):
    # Читаем данные
    df_edges = pd.read_csv(osp.join(data_path, "elliptic_txs_edgelist.csv"))
    df_features = pd.read_csv(osp.join(data_path, "tx_features_robust_base_tx_wallet_scaled.csv"))
    df_classes = pd.read_csv(osp.join(data_path, "elliptic_txs_classes.csv"))

    # Меняем класс 'unknown' на '3' (или исключаем — можно выбрать)
    df_classes.loc[df_classes['class'] == 'unknown', 'class'] = '3'

    # Приводим txId к строкам для корректного объединения
    df_classes["txId"] = df_classes["txId"].astype(str)
    df_features["txId"] = df_features["txId"].astype(str)
    df_edges["txId1"] = df_edges["txId1"].astype(str)
    df_edges["txId2"] = df_edges["txId2"].astype(str)

    # Объединяем фичи и классы по txId
    df_class_feature = pd.merge(df_classes, df_features, how="inner", on="txId")

    # Исключаем неизвестные классы
    df_class_feature = df_class_feature[df_class_feature["class"] != '3']

    # Фильтруем рёбра — оставляем только ребра между известными транзакциями
    known_txs = set(df_class_feature["txId"])
    df_edges = df_edges[(df_edges["txId1"].isin(known_txs)) & (df_edges["txId2"].isin(known_txs))]

    # Создаем индексы для txId и class
    features_idx = {tx: idx for idx, tx in enumerate(sorted(df_class_feature["txId"].unique()))}
    class_idx = {cls: idx for idx, cls in enumerate(sorted(df_class_feature["class"].unique()))}

    # Кодируем txId и class в индексы
    df_class_feature["txId"] = df_class_feature["txId"].map(features_idx)
    df_class_feature["class"] = df_class_feature["class"].map(class_idx)
    df_edges["txId1"] = df_edges["txId1"].map(features_idx)
    df_edges["txId2"] = df_edges["txId2"].map(features_idx)

    return df_class_feature, df_edges

def data_to_pyg(df_class_feature, df_edges):
    features = df_class_feature.iloc[:, 3:].apply(pd.to_numeric, errors='coerce').fillna(0).values
    edge_index = torch.tensor([df_edges["txId1"].values, df_edges["txId2"].values], dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(df_class_feature["class"].values, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    data = RandomNodeSplit(num_val=0.15, num_test=0.2)(data)

    return data