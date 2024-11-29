import warnings
import torch
import matplotlib.pyplot as plt
import pandas as pd
import utils as u
import os
from loader import load_data, data_to_pyg
from train import train, test
from models import models

# ...

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

data_path = '.\data\elliptic\dataset'

print("Loading configuration from file...")
args = u.get_config()
print("Configuration loaded successfully")
print("="*50)
print("Loading graph data...")
data_path = args.data_path if data_path is None else data_path

features, edges = load_data(data_path)
features_noAgg, edges_noAgg = load_data(data_path, noAgg=True)

u.seed_everything(42)

data = data_to_pyg(features, edges)
data_noAgg = data_to_pyg(features_noAgg, edges_noAgg)

print("Graph data loaded successfully")
print("="*50)
args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
args.device = 'cpu'
if args.use_cuda:
    args.device = 'cuda'
print ("Using CUDA: ", args.use_cuda, "- args.device: ", args.device)

models_to_train = {
    'Chebyshev Convolution (tx)': models.ChebyshevConvolution(args, [1, 2, 3, 4], data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'Chebyshev Convolution (tx+agg)': models.ChebyshevConvolution(args, [1, 2, 3, 4], data.num_features, args.hidden_units).to(args.device),
}

compare_illicit = pd.DataFrame(columns=['model','Precision','Recall', 'F1', 'F1 Micro AVG'])
print("Starting training models")
print("="*50)

model_list = list(models_to_train.items())


def plot_metrics(metrics):
    epochs = metrics['epoch']

    plt.figure(figsize=(12, 8))

    # Subplot for Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics['train_loss'], label='Train Loss')
    plt.plot(epochs, metrics['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    # Subplot for Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics['train_acc'], label='Train Acc')
    plt.plot(epochs, metrics['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    # Subplot for Precision, Recall, F1
    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics['precision'], label='Precision')
    plt.plot(epochs, metrics['recall'], label='Recall')
    plt.plot(epochs, metrics['f1'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision, Recall, F1 Over Epochs')
    plt.legend()

    # Subplot for F1 Micro AVG
    plt.subplot(2, 2, 4)
    plt.plot(epochs, metrics['f1_micro'], label='F1 Micro AVG')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Micro AVG')
    plt.title('F1 Micro AVG Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

for i in range(0, len(model_list), 2):
    # Call train function
    (name, model) = model_list[i]
    _, metrics_over_epochs = train(args, model, data)
    plot_metrics(metrics_over_epochs)

    # (name, model) = model_list[i + 1]
    # data = data.to(args.device)
    # print('-'*50)
    # print(f"Training model: {name}")
    # print('-'*50)
    # train(args, model, data)
    # print('-'*50)
    # print(f"Testing model: {name}")
    # print('-'*50)
    # test(model, data)
    # print('-'*50)
    # print(f"Computing metrics for model: {name}")
    # compare_illicit = pd.concat(
    #     [compare_illicit, pd.DataFrame([u.compute_metrics(model, name, data, compare_illicit)])],
    #     ignore_index=True)
    # print('-'*50)
    

compare_illicit.to_csv(os.path.join('.\output', 'metrics_MultiStepLR.csv'), index=False)
print('Results saved to metrics.csv')

u.plot_results(compare_illicit)

u.aggregate_plot(compare_illicit)

