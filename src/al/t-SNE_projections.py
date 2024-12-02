from functions.supervised_experiments import supervised_prediction, calculate_metrics
from functions.visualizations import plot_TSNE_projection
from functions.elliptic_dataset_preprocessing import preprocessing_pipeline, split_data
from functions.config import features_path, classes_path, edges_path

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


train_data, test_data = preprocessing_pipeline(last_train_timestep=34, last_test_timestep=49, only_labeled = True, features_path=features_path, classes_path=classes_path, edges_path=edges_path)
X_train, y_train, X_test, y_test = split_data(train_data = train_data, test_data = test_data)

#Plot t-SNE projections

#plot true labels
reducer = TSNE(n_components=2, init='pca', learning_rate='auto', n_iter=10000)
embedding = reducer.fit_transform(X_test)

embedding_df = pd.DataFrame(embedding, columns=('dim_0', 'dim_1'))
embedding_df['class'] = y_test.tolist()
embedding_df['class'] = embedding_df['class'].replace({1: 'Illicit', 0: 'Licit'})

plot_TSNE_projection(title='True labels', embedding_df=embedding_df, hue_on='class', fontsize=13, labelsize=15,
                     palette=['green', 'red'])


#plot the Random Forest predicted labels
model_rf = RandomForestClassifier()
probabilities_rf = supervised_prediction(X_train, y_train, X_test, model_rf, runs=1)
(_, _, _, max_threshold_rf) = calculate_metrics(y_test, probabilities_rf)
# Extract the scalar threshold from the list
max_threshold_rf = max_threshold_rf[0]
# Ensure probabilities_rf is a 2D array
print(f"Probabilities: {probabilities_rf}")

# Apply the threshold to the second column of probabilities
predictions_rf = (probabilities_rf[0][:, 1] > max_threshold_rf).astype(int)

reducer = TSNE(n_components=2, init='pca', learning_rate='auto', n_iter=10000)
embedding_rf = reducer.fit_transform(X_test)

embedding_rf_df = pd.DataFrame(embedding_rf, columns=('dim_0', 'dim_1'))
embedding_rf_df['prediction'] = ['Illicit' if pred == 1 else 'Licit' for pred in
                              predictions_rf]


plot_TSNE_projection(title='Random Forest predictions', embedding_df=embedding_rf_df, hue_on='prediction', fontsize=13, labelsize=15,
                     palette=['green', 'red'])


# plot the XGBoost predicted labels
model_xgb = XGBClassifier()
probabilities_xgb = supervised_prediction(X_train, y_train, X_test, model_xgb, runs=1)
(_,_, _, max_threshold_xgb) = calculate_metrics(y_test, probabilities_xgb)
predictions_xgb = (probabilities_xgb[0][:, 1] > max_threshold_xgb).astype(int)

reducer = TSNE(n_components=2, init='pca', learning_rate='auto', n_iter=10000)
embedding_xgb = reducer.fit_transform(X_test)

embedding_xgb_df = pd.DataFrame(embedding_xgb, columns=('dim_0', 'dim_1'))
embedding_xgb_df['prediction'] = ['Illicit' if pred == 1 else 'Licit' for pred in
                              predictions_xgb]


plot_TSNE_projection(title='XGBoost predictions', embedding_df=embedding_xgb_df, hue_on='prediction', fontsize=13, labelsize=15,
                     palette=['green', 'red'])


#plot the Logistic Regression predicted labels
model_lr = LogisticRegression(max_iter=10000)
probabilities_lr = supervised_prediction(X_train, y_train, X_test, model_lr, runs=1)
(_,_, _, max_threshold_lr) = calculate_metrics(y_test, probabilities_lr)
predictions_lr = (probabilities_lr[0][:, 1] > max_threshold_lr).astype(int)

reducer = TSNE(n_components=2, init='pca', learning_rate='auto', n_iter=10000)
embedding_lr = reducer.fit_transform(X_test)

embedding_lr_df = pd.DataFrame(embedding_lr, columns=('dim_0', 'dim_1'))
embedding_lr_df['prediction'] = ['Illicit' if pred == 1 else 'Licit' for pred in
                              predictions_lr]


plot_TSNE_projection(title='Logistic Regression predictions', embedding_df=embedding_lr_df, hue_on='prediction', fontsize=13, labelsize=15,
                     palette=['green', 'red'])


# Plot true labels using PCA
pca = PCA(n_components=2)
embedding_pca = pca.fit_transform(X_test)

embedding_df_pca = pd.DataFrame(embedding_pca, columns=('dim_0', 'dim_1'))
embedding_df_pca['class'] = y_test.tolist()
embedding_df_pca['class'] = embedding_df_pca['class'].replace({1: 'Illicit', 0: 'Licit'})

plot_TSNE_projection(title='True labels (PCA)', embedding_df=embedding_df_pca, hue_on='class', fontsize=13, labelsize=15,
                     palette=['green', 'red'])

# Plot Random Forest predicted labels using PCA
embedding_rf_pca = pca.fit_transform(X_test)
embedding_rf_pca_df = pd.DataFrame(embedding_rf_pca, columns=('dim_0', 'dim_1'))
embedding_rf_pca_df['prediction'] = ['Illicit' if pred == 1 else 'Licit' for pred in predictions_rf]

plot_TSNE_projection(title='Random Forest predictions (PCA)', embedding_df=embedding_rf_pca_df, hue_on='prediction', fontsize=13, labelsize=15,
                     palette=['green', 'red'])

# Plot XGBoost predicted labels using PCA
embedding_xgb_pca = pca.fit_transform(X_test)
embedding_xgb_pca_df = pd.DataFrame(embedding_xgb_pca, columns=('dim_0', 'dim_1'))
embedding_xgb_pca_df['prediction'] = ['Illicit' if pred == 1 else 'Licit' for pred in predictions_xgb]

plot_TSNE_projection(title='XGBoost predictions (PCA)', embedding_df=embedding_xgb_pca_df, hue_on='prediction', fontsize=13, labelsize=15,
                     palette=['green', 'red'])

# Plot Logistic Regression predicted labels using PCA
embedding_lr_pca = pca.fit_transform(X_test)
embedding_lr_pca_df = pd.DataFrame(embedding_lr_pca, columns=('dim_0', 'dim_1'))
embedding_lr_pca_df['prediction'] = ['Illicit' if pred == 1 else 'Licit' for pred in predictions_lr]

plot_TSNE_projection(title='Logistic Regression predictions (PCA)', embedding_df=embedding_lr_pca_df, hue_on='prediction', fontsize=13, labelsize=15,
                     palette=['green', 'red'])
