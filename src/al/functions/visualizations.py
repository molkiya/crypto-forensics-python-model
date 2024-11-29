from functions.elliptic_dataset_preprocessing import get_data, split_data

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score, f1_score, \
    average_precision_score


# Plot f1-score by timestep (Translated labels)
def plot_f1_by_timestep(model_metric_dict, last_train_time_step=34, last_time_step=49, fontsize=15, labelsize=18,
                        figsize=(20, 10),
                        linestyle=['solid', "dotted", 'dashed'], linecolor=["green", "orange", "red"],
                        barcolor='lightgrey', baralpha=0.3, linewidth=1.5):
    data = get_data(only_labeled=True)
    occ = data.groupby(['timestep', 'class']).size().to_frame(name='occurences').reset_index()
    illicit_per_timestep = occ[(occ['class'] == 1) & (occ['timestep'] > 34)]

    timesteps = illicit_per_timestep['timestep'].unique()
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    i = 0
    for key, values in model_metric_dict.items():
        if key != "XGBoost":
            key = key.lower()
        ax1.plot(timesteps, values, label=key, linestyle=linestyle[i], color=linecolor[i], linewidth=linewidth)
        i += 1

    ax2.bar(timesteps, illicit_per_timestep['occurences'], color=barcolor, alpha=baralpha,
            label='Незаконные транзакции')
    ax2.get_yaxis().set_visible(True)
    ax2.tick_params(axis='both', which='major', labelsize=labelsize)
    ax2.grid(False)

    ax1.set_xlabel('Шаг времени', fontsize=fontsize)
    ax1.set_ylabel('F1-Оценка', fontsize=fontsize)
    ax1.set_xticks(range(last_train_time_step + 1, last_time_step + 1))
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax1.tick_params(axis='both', which='major', labelsize=labelsize)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    ax1.legend(lines, labels, fontsize=fontsize, facecolor="#EEEEEE")

    ax1.tick_params(direction='in')

    ax2.set_ylabel('Кол-во незаконных транзакций', fontsize=fontsize)

    plt.title('F1-Оценка по шагам времени')

    return fig


# Plot metrics by threshold (Translated labels + additional metrics)
def draw_metrics_by_threshold(models, X_train, y_train, X_test, y_test):
    for name, model in models.items():
        print(f"Запуск {name}")
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)

        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob[:, 1])
        f1_scores = [f1_score(y_test, y_pred_prob[:, 1] > threshold, pos_label=1) for threshold in thresholds]

        plt.plot(thresholds, precision[:-1], label='Точность')
        plt.plot(thresholds, recall[:-1], label='Полнота')
        plt.plot(thresholds, f1_scores, label='F1-Оценка')

        plt.xlabel('Порог')
        plt.ylabel('F1-Оценка')

        max_f1_index = np.argmax(f1_scores)
        max_f1_threshold = thresholds[max_f1_index]
        max_f1_score = f1_scores[max_f1_index]

        plt.axvline(x=max_f1_threshold, color='black', linestyle='--')
        plt.text(max_f1_threshold + 0.01, max_f1_score - 0.1, f'Макс. F1-Оценка = {max_f1_score:.2f}', rotation=90)

        # Calculate and display additional metrics (Precision, Recall, F1, and Average Precision)
        precision_val = precision_score(y_test, y_pred_prob[:, 1] > max_f1_threshold)
        recall_val = recall_score(y_test, y_pred_prob[:, 1] > max_f1_threshold)
        f1_val = f1_score(y_test, y_pred_prob[:, 1] > max_f1_threshold)
        avg_precision = average_precision_score(y_test, y_pred_prob[:, 1])

        plt.title(
            f'{name} (Точность: {precision_val:.2f}, Полнота: {recall_val:.2f}, F1: {f1_val:.2f}, Ср. точность: {avg_precision:.2f})')
        plt.legend()
        plt.show()


# Plot TSNE (Translated labels)
def plot_TSNE_projection(title, embedding_df, hue_on, fontsize, labelsize, palette, linewidth=0.000001,
                         savefig_path=None):
    fig_dims = (10, 6)
    fig, ax = plt.subplots(figsize=fig_dims)
    sns.scatterplot(x='dim_0', y='dim_1', hue=hue_on, style=hue_on, markers=['.', 'X'], size=hue_on,
                    sizes=[150, 170], linewidth=linewidth, palette=palette, data=embedding_df, ax=ax)
    ax.set_xlabel('Размерность 1', fontsize=fontsize)
    ax.set_ylabel('Размерность 2', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize, direction='in')
    L = ax.legend(prop={'size': labelsize}, facecolor="#EEEEEE", handletextpad=-0.5)

    L.get_texts()[0].set_text("Легальная")
    L.get_texts()[1].set_text("Незаконная")

    ax.set_title(title, fontsize=fontsize)

    if savefig_path == None:
        plt.show()
    else:
        plt.savefig(savefig_path, bbox_inches='tight', pad_inches=0)


# Learning Curves (Translated labels)
def plot_learning_curves(al_stats_df, alpha=0.2, figsize=(10, 8)):
    plt.figure(figsize=figsize)

    # Get the model names
    model_names = al_stats_df['supervised classifier'].unique()

    for model_name in model_names:
        # Get the values for the specific supervised classifier
        data = al_stats_df[al_stats_df['supervised classifier'] == model_name]
        f1_scores_mean = data['f1-score'].values
        # f1_scores_std = data['std'].values
        train_sizes = data['labeled pool size'].values

        plt.plot(train_sizes, f1_scores_mean, '--', label=model_name)

        plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000])
        plt.yticks([0, 0.25, 0.5, 0.75, 1.0])

    plt.axhline(y=0.523, color='black', linestyle='--')
    plt.text(x=200, y=0.53, s='0.52 (Логистическая регрессия)', color='black', ha='left', va='center')

    if (al_stats_df['hot-learner'].unique()[0] != ''):
        al_setup = al_stats_df['warm-up learner'].unique()[0] + ' и ' + al_stats_df['hot-learner'].unique()[0]

    # Set the plot parameters
    plt.xlabel('Размер обучающей выборки')
    plt.ylabel('F1-Оценка')
    plt.title(al_setup)
    plt.legend()
    plt.show()
