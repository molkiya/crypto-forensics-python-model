"""
Адаптер для преобразования JSON запроса от Rust в формат графа для модели
"""
import torch
from torch_geometric.data import Data
from typing import List
from .models import MLRequest


def create_graph_from_request(request: MLRequest) -> Data:
    """
    Преобразует JSON запрос от Rust в графовую структуру для модели.
    
    Создает граф, где:
    - Центральный узел: транзакция
    - Остальные узлы: адреса входов и выходов
    - Рёбра: связи между транзакцией и адресами
    
    Args:
        request: Запрос от Rust приложения
        
    Returns:
        Data: Граф в формате PyTorch Geometric
    """
    # Собираем все признаки для узлов
    nodes = []
    node_types = []  # 0 = транзакция, 1 = входной адрес, 2 = выходной адрес
    
    # 1. Центральный узел - транзакция
    tx_features = _extract_transaction_features(request)
    nodes.append(tx_features)
    node_types.append(0)
    
    # 2. Узлы входных адресов
    input_start_idx = len(nodes)
    input_count = 0
    for addr_feat in request.input_features or []:
        # Используем признаки адреса, если они есть
        if addr_feat.features and len(addr_feat.features) > 0:
            # Берем первые признаки (может быть меньше 55, дополняем нулями)
            features = addr_feat.features[:55] if len(addr_feat.features) >= 55 else addr_feat.features + [0.0] * (55 - len(addr_feat.features))
            nodes.append(features)
        else:
            # Если признаков нет, создаем из базовых характеристик
            features = _create_address_features_from_tx(request.transaction_features, is_input=True)
            nodes.append(features)
        node_types.append(1)
        input_count += 1
    
    # 3. Узлы выходных адресов
    output_start_idx = len(nodes)
    for addr_feat in request.output_features or []:
        if addr_feat.features and len(addr_feat.features) > 0:
            features = addr_feat.features[:55] if len(addr_feat.features) >= 55 else addr_feat.features + [0.0] * (55 - len(addr_feat.features))
            nodes.append(features)
        else:
            features = _create_address_features_from_tx(request.transaction_features, is_input=False)
            nodes.append(features)
        node_types.append(2)
    
    # Если адресов нет, создаем фиктивные узлы на основе базовых признаков
    if len(nodes) == 1:  # Только транзакция
        # Создаем по одному узлу для каждого входа и выхода
        for i in range(request.transaction_features.n_inputs):
            features = _create_address_features_from_tx(request.transaction_features, is_input=True)
            nodes.append(features)
            node_types.append(1)
        
        for i in range(request.transaction_features.n_outputs):
            features = _create_address_features_from_tx(request.transaction_features, is_input=False)
            nodes.append(features)
            node_types.append(2)
    
    # Преобразуем в тензор
    x = torch.tensor(nodes, dtype=torch.float32)
    
    # Создаем рёбра: транзакция связана со всеми адресами
    edges = []
    tx_idx = 0
    
    # Рёбра от транзакции к входным адресам
    if input_start_idx < output_start_idx:
        for i in range(input_start_idx, output_start_idx):
            edges.append([tx_idx, i])
            edges.append([i, tx_idx])  # Неориентированный граф
    
    # Рёбра от транзакции к выходным адресам
    if output_start_idx < len(nodes):
        for i in range(output_start_idx, len(nodes)):
            edges.append([tx_idx, i])
            edges.append([i, tx_idx])  # Неориентированный граф
    
    if not edges:
        # Если нет рёбер, создаем самосвязь для транзакции
        edges = [[0, 0]]
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Создаем объект Data
    data = Data(x=x, edge_index=edge_index)
    
    return data


def _extract_transaction_features(request: MLRequest) -> List[float]:
    """
    Извлекает признаки транзакции из запроса.
    
    Создает вектор признаков, совместимый с форматом модели.
    """
    tf = request.transaction_features
    ef = request.extended_features
    
    features = []
    
    # Базовые признаки транзакции (7 признаков)
    features.extend([
        float(tf.n_inputs),
        float(tf.n_outputs),
        tf.input_value_sum,
        tf.output_value_sum,
        tf.transaction_fee,
        tf.avg_input_value,
        tf.avg_output_value,
    ])
    
    # Расширенные признаки (если есть)
    if ef:
        features.extend([
            float(ef.time_step) if ef.time_step is not None else 0.0,
            ef.avg_input_incoming_txs or 0.0,
            ef.avg_output_outgoing_txs or 0.0,
            float(ef.unique_input_addresses) if ef.unique_input_addresses is not None else 0.0,
            float(ef.unique_output_addresses) if ef.unique_output_addresses is not None else 0.0,
            float(ef.num_coinbase_inputs) if ef.num_coinbase_inputs is not None else 0.0,
            ef.old_input_fraction or 0.0,
            ef.change_output_ratio or 0.0,
            ef.inputs_address_entropy or 0.0,
            ef.outputs_address_entropy or 0.0,
            float(ef.spent_outputs_count) if ef.spent_outputs_count is not None else 0.0,
            float(ef.unspent_outputs_count) if ef.unspent_outputs_count is not None else 0.0,
            ef.time_diff_prev_output or 0.0,
            ef.avg_outgoing_txs_inputs or 0.0,
            ef.avg_incoming_txs_outputs or 0.0,
        ])
    else:
        # Заполняем нулями, если расширенных признаков нет
        features.extend([0.0] * 15)
    
    # Дополняем до нужного размера (например, 94 признака как в оригинале)
    # Если нужно больше признаков, можно добавить статистики по адресам
    target_size = 94  # Размер признаков из оригинального датасета
    while len(features) < target_size:
        features.append(0.0)
    
    # Обрезаем, если больше
    features = features[:target_size]
    
    return features


def _create_address_features_from_tx(tx_features: 'TransactionFeatures', is_input: bool) -> List[float]:
    """
    Создает признаки адреса на основе признаков транзакции.
    Используется, когда признаки адреса не предоставлены.
    """
    # Базовые признаки адреса (55 признаков)
    # Используем пропорциональные значения от транзакции
    if is_input:
        base_value = tx_features.avg_input_value
        count = tx_features.n_inputs
    else:
        base_value = tx_features.avg_output_value
        count = tx_features.n_outputs
    
    # Создаем упрощенные признаки адреса
    features = [
        float(count),  # time_step заменяем на количество
        1.0 if is_input else 0.0,  # флаг входа/выхода
        0.0,  # n_outputs для адреса
        base_value,  # input_value_sum для адреса
        base_value,  # output_value_sum для адреса
        0.0,  # transaction_fee
        base_value,  # avg_input_value
        base_value,  # avg_output_value
    ]
    
    # Дополняем до 55 признаков
    while len(features) < 55:
        features.append(0.0)
    
    return features[:55]

