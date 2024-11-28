# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df_classes = pd.read_csv("data/elliptic/dataset/elliptic_txs_classes.csv")
df_edges = pd.read_csv("data/elliptic/dataset/elliptic_txs_edgelist.csv")
df_features = pd.read_csv("data/elliptic/dataset/elliptic_txs_features.csv", header=None)
colNames1 = {'0': 'txId', 1: "Time step"}
colNames2 = {str(ii+2): "Local_feature_" + str(ii+1) for ii in range(93)}
colNames3 = {str(ii+95): "Aggregate_feature_" + str(ii+1) for ii in range(72)}

colNames = dict(colNames1, **colNames2, **colNames3 )
colNames = {int(jj): item_kk for jj,item_kk in colNames.items()}
df_features = df_features.rename(columns=colNames)

print(df_features.head())

df_classes.loc[df_classes['class'] == 'unknown', 'class'] = 3
print('Shape of classes', df_classes.shape)
print('Shape of edges', df_edges.shape)
print('Shape of features', df_features.shape)

group_class = df_classes.groupby('class').count()
plt.barh(['Неизвестные', 'Нелегальные', 'Легальные'], group_class['txId'].values, color=['orange', 'r', 'g'])
plt.show()

group_feature = df_features.groupby('Time step').count()
group_feature['txId'].plot()
plt.title('Number of transactions by Time step')
plt.show()

df_class_feature = pd.merge(df_classes, df_features)
print(df_class_feature.head())

group_class_feature = df_class_feature.groupby(['Time step', 'class']).count()
group_class_feature = group_class_feature['txId'].reset_index().rename(columns={'txId': 'count'})
print(group_class_feature.head())

sns.lineplot(x='Time step', y='count', hue='class', data = group_class_feature, palette=['g', 'orange', 'r'] )
plt.show()

class1 = group_class_feature[group_class_feature['class'] == '1']
class2 = group_class_feature[group_class_feature['class'] == '2']
class3 = group_class_feature[group_class_feature['class'] == 3 ]

p1 = plt.bar(class3['Time step'], class3['count'], color = 'orange')

p2 = plt.bar(class2['Time step'], class2['count'], color='g',
             bottom=class3['count'])

p3 = plt.bar(class1['Time step'], class1['count'], color='r',
             bottom=np.array(class3['count'])+np.array(class2['count']))

plt.xlabel('Time step')
plt.show()

# Filter illicit transactions for Time Step 20
illicit_ids = df_class_feature.loc[(df_class_feature['Time step'] == 20) & (df_class_feature['class'] == '1'), 'txId']
illicit_edges = df_edges.loc[df_edges['txId1'].isin(illicit_ids)]

# Create directed graph
graph_illicit = nx.from_pandas_edgelist(illicit_edges, source='txId1', target='txId2', create_using=nx.DiGraph())

# Add node and edge details
pos_illicit = nx.spring_layout(graph_illicit)
plt.figure(figsize=(12, 8))
nx.draw(graph_illicit, pos=pos_illicit, with_labels=True, node_size=100, node_color='red', edge_color='blue', alpha=0.7)
plt.title("Illicit Transactions - Time Step 20")
plt.show()

# Filter licit transactions for Time Step 20
licit_ids = df_class_feature.loc[(df_class_feature['Time step'] == 20) & (df_class_feature['class'] == '2'), 'txId']
licit_edges = df_edges.loc[df_edges['txId1'].isin(licit_ids)]

# Create directed graph
graph_licit = nx.from_pandas_edgelist(licit_edges, source='txId1', target='txId2', create_using=nx.DiGraph())

# Add node and edge details
pos_licit = nx.spring_layout(graph_licit)
plt.figure(figsize=(12, 8))
nx.draw(graph_licit, pos=pos_licit, with_labels=True, node_size=100, node_color='green', edge_color='orange', alpha=0.7)
plt.title("Licit Transactions - Time Step 20")
plt.show()

# Filter illicit transactions for Time Step 37
bad_ids = df_features.loc[(df_features['time step'] == 37) & (df_features['class'] == '1'), 'id']
short_edges = df_edgelist.loc[df_edgelist['txId1'].isin(bad_ids)]

# Create directed graph
graph_bad = nx.from_pandas_edgelist(short_edges, source='txId1', target='txId2', create_using=nx.DiGraph())

# Add node and edge details
pos_bad = nx.spring_layout(graph_bad)
plt.figure(figsize=(12, 8))
nx.draw(graph_bad, pos=pos_bad, cmap=plt.get_cmap('rainbow'), with_labels=True, node_size=200, edge_color='gray', alpha=0.8)
plt.title("Illicit Transactions - Time Step 37")
plt.show()

# Prediction of ilicit transactions

selected_ids = df_class_feature.loc[(df_class_feature['class'] != 3), 'txId']
df_edges_selected = df_edges.loc[df_edges['txId1'].isin(selected_ids)]
df_classes_selected = df_classes.loc[df_classes['txId'].isin(selected_ids)]
df_features_selected = df_features.loc[df_features['txId'].isin(selected_ids)]

# Merge Class and features
df_class_feature_selected = pd.merge(df_classes_selected, df_features_selected )
df_class_feature_selected.head()
X = df_class_feature_selected.drop(columns=['txId', 'class', 'Time step']) # drop class, text id and time step
y = df_class_feature_selected[['class']]

# in this case, class 2 corresponds to licit transactions, we chang this to 0 as our interest is the ilicit transactions
y = y['class'].apply(lambda x: 0 if x == '2' else 1 )

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=15)
model_RF = RandomForestClassifier().fit(X_train.values,y_train.values)
y_preds = model_RF.predict(X_test.values)

prec,rec,f1,num = precision_recall_fscore_support(y_test.values, y_preds)

print("Random Forest Classifier")
print("Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f"%(prec[1],rec[1],f1[1]))
unknown_ids = df_class_feature.loc[(df_class_feature['class'] == 3), 'txId']

df_edges_unknown = df_edges.loc[df_edges['txId1'].isin(unknown_ids)]
df_classes_unknown = df_classes.loc[df_classes['txId'].isin(unknown_ids)]
df_features_unknown = df_features.loc[df_features['txId'].isin(unknown_ids)]

X_unknown = df_features_unknown.drop(columns=['txId', 'Time step'])
y_unknown_preds = model_RF.predict(X_unknown.values)

df_classes_unknown = df_classes_unknown.copy()
df_classes_unknown.loc[:, 'class'] = y_unknown_preds

df_class_feature_unknown = pd.merge(df_classes_unknown, df_features_unknown )
df_class_feature_unknown.head()

# Predicted Graph
ilicit_ids = df_class_feature_unknown.loc[(df_class_feature_unknown['Time step'] == 20) & (df_class_feature_unknown['class'] == 1), 'txId']
ilicit_edges = df_edges_unknown.loc[df_edges_unknown['txId1'].isin(ilicit_ids)]

graph = nx.from_pandas_edgelist(ilicit_edges, source = 'txId1', target = 'txId2',
                                 create_using = nx.DiGraph())
pos = nx.spring_layout(graph)
nx.draw(graph, with_labels=False, pos=pos)