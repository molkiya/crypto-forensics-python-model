import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

# Проверка файлов в директории
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Загрузка данных
df_classes = pd.read_csv("data/elliptic/dataset/elliptic_txs_classes.csv")
df_edges = pd.read_csv("data/elliptic/dataset/elliptic_txs_edgelist.csv")
df_features = pd.read_csv("data/elliptic/dataset/elliptic_txs_features.csv", header=None)

# Переименование колонок
colNames1 = {'0': 'txId', 1: "Time step"}
colNames2 = {str(ii+2): "Local_feature_" + str(ii+1) for ii in range(93)}
colNames3 = {str(ii+95): "Aggregate_feature_" + str(ii+1) for ii in range(72)}

colNames = dict(colNames1, **colNames2, **colNames3 )
colNames = {int(jj): item_kk for jj,item_kk in colNames.items()}
df_features = df_features.rename(columns=colNames)

print("Первые строки фичей:")
print(df_features.head())

# Заменяем 'unknown' на 3
df_classes.loc[df_classes['class'] == 'unknown', 'class'] = 3
print('Размер классов:', df_classes.shape)
print('Размер рёбер:', df_edges.shape)
print('Размер фичей:', df_features.shape)

# Построение диаграммы количества транзакций по классам
group_class = df_classes.groupby('class').count()
plt.barh(['Неизвестные', 'Нелегальные', 'Легальные'], group_class['txId'].values, color=['orange', 'r', 'g'])
plt.title('Количество транзакций по классам')
plt.xlabel('Количество транзакций')
plt.ylabel('Классы')
plt.show()

# График количества транзакций по временным шагам
group_feature = df_features.groupby('Time step').count()
group_feature['txId'].plot()
plt.title('Количество транзакций по временным шагам')
plt.xlabel('Временные шаги')
plt.ylabel('Количество транзакций')
plt.show()

# Объединение классов и фичей
df_class_feature = pd.merge(df_classes, df_features)
print("Первые строки объединённых данных:")
print(df_class_feature.head())

# Группировка по временным шагам и классам
group_class_feature = df_class_feature.groupby(['Time step', 'class']).count()
group_class_feature = group_class_feature['txId'].reset_index().rename(columns={'txId': 'count'})
print("Группировка по временным шагам и классам:")
print(group_class_feature.head())

# Линейный график количества транзакций по классам
sns.lineplot(x='Time step', y='count', hue='class', data=group_class_feature, palette=['g', 'orange', 'r'])
plt.title('Количество транзакций по классам и временным шагам')
plt.xlabel('Временные шаги')
plt.ylabel('Количество транзакций')
plt.show()

# Стековая диаграмма по классам
class1 = group_class_feature[group_class_feature['class'] == '1']
class2 = group_class_feature[group_class_feature['class'] == '2']
class3 = group_class_feature[group_class_feature['class'] == 3]

p1 = plt.bar(class3['Time step'], class3['count'], color='orange', label='Неизвестные')
p2 = plt.bar(class2['Time step'], class2['count'], color='g', bottom=class3['count'], label='Легальные')
p3 = plt.bar(class1['Time step'], class1['count'], color='r', bottom=np.array(class3['count']) + np.array(class2['count']), label='Нелегальные')

plt.xlabel('Временные шаги')
plt.ylabel('Количество транзакций')
plt.title('Стековая диаграмма транзакций')
plt.legend()
plt.show()

# Анализ нелегальных транзакций на временном шаге 20
illicit_ids = df_class_feature.loc[(df_class_feature['Time step'] == 20) & (df_class_feature['class'] == '1'), 'txId']
illicit_edges = df_edges.loc[df_edges['txId1'].isin(illicit_ids)]

graph_illicit = nx.from_pandas_edgelist(illicit_edges, source='txId1', target='txId2', create_using=nx.DiGraph())
pos_illicit = nx.spring_layout(graph_illicit)
plt.figure(figsize=(12, 8))
nx.draw(graph_illicit, pos=pos_illicit, with_labels=True, node_size=100, node_color='red', edge_color='blue', alpha=0.7)
plt.title("Граф нелегальных транзакций - Временной шаг 20")
plt.show()

# Анализ легальных транзакций на временном шаге 20
licit_ids = df_class_feature.loc[(df_class_feature['Time step'] == 20) & (df_class_feature['class'] == '2'), 'txId']
licit_edges = df_edges.loc[df_edges['txId1'].isin(licit_ids)]

graph_licit = nx.from_pandas_edgelist(licit_edges, source='txId1', target='txId2', create_using=nx.DiGraph())
pos_licit = nx.spring_layout(graph_licit)
plt.figure(figsize=(12, 8))
nx.draw(graph_licit, pos=pos_licit, with_labels=True, node_size=100, node_color='green', edge_color='orange', alpha=0.7)
plt.title("Граф легальных транзакций - Временной шаг 20")
plt.show()

# Предсказание нелегальных транзакций
selected_ids = df_class_feature.loc[(df_class_feature['class'] != 3), 'txId']
df_edges_selected = df_edges.loc[df_edges['txId1'].isin(selected_ids)]
df_classes_selected = df_classes.loc[df_classes['txId'].isin(selected_ids)]
df_features_selected = df_features.loc[df_features['txId'].isin(selected_ids)]

df_class_feature_selected = pd.merge(df_classes_selected, df_features_selected)
X = df_class_feature_selected.drop(columns=['txId', 'class', 'Time step'])
y = df_class_feature_selected['class'].apply(lambda x: 0 if x == '2' else 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
model_RF = RandomForestClassifier().fit(X_train.values, y_train.values)
y_preds = model_RF.predict(X_test.values)

prec, rec, f1, _ = precision_recall_fscore_support(y_test.values, y_preds)

print("Random Forest Classifier")
print("Точность: %.3f" % prec[1])
print("Полнота: %.3f" % rec[1])
print("F1-мера: %.3f" % f1[1])

unknown_ids = df_class_feature.loc[(df_class_feature['class'] == 3), 'txId']
df_edges_unknown = df_edges.loc[df_edges['txId1'].isin(unknown_ids)]
df_classes_unknown = df_classes.loc[df_classes['txId'].isin(unknown_ids)]
df_features_unknown = df_features.loc[df_features['txId'].isin(unknown_ids)]

X_unknown = df_features_unknown.drop(columns=['txId', 'Time step'])
y_unknown_preds = model_RF.predict(X_unknown.values)

df_classes_unknown = df_classes_unknown.copy()
df_classes_unknown['class'] = y_unknown_preds

df_class_feature_unknown = pd.merge(df_classes_unknown, df_features_unknown)

# Граф предсказанных нелегальных транзакций
illicit_ids = df_class_feature_unknown.loc[(df_class_feature_unknown['Time step'] == 20) & (df_class_feature_unknown['class'] == 1), 'txId']
illicit_edges = df_edges_unknown.loc[df_edges_unknown['txId1'].isin(illicit_ids)]

graph_predicted = nx.from_pandas_edgelist(illicit_edges, source='txId1', target='txId2', create_using=nx.DiGraph())
pos_predicted = nx.spring_layout(graph_predicted)
plt.figure(figsize=(12, 8))
nx.draw(graph_predicted, pos=pos_predicted, with_labels=True, node_size=100, node_color='purple', edge_color='gray', alpha=0.7)
plt.title("Предсказанные нелегальные транзакции - Временной шаг 20")
plt.show()
