import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


def train(args, model, data):
    """Train a GNN model, return the trained model, and plot metrics."""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    else:
        print("No GPU available. Training will run on CPU.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    optimizer = Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.9, verbose=True)
    epochs = args['epochs']
    model.train()

    best_val_loss = float('inf')
    patience = 1000
    epochs_since_best = 0

    # Metrics storage
    metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'f1_micro': []
    }

    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        out, _ = model((data.x, data.edge_index))
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        with torch.no_grad():
            val_out = out[data.val_mask].argmax(dim=1)
            val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
            val_acc = accuracy(val_out, data.y[data.val_mask])

            # Adjust learning rate
            scheduler.step(val_loss)

            # Compute metrics
            y_true = data.y[data.val_mask].cpu().numpy()
            y_pred = val_out.cpu().numpy()

            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

            # Store metrics
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            metrics['f1_micro'].append(f1_micro)

        # Check if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        # Print metrics every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                  f'{acc * 100:>6.2f}% | Val Loss: {val_loss:.3f} | '
                  f'Val Acc: {val_acc * 100:.2f}% | Precision: {precision:.3f} | '
                  f'Recall: {recall:.3f} | F1: {f1:.3f} | F1 (Micro): {f1_micro:.3f}')

        # Check if early stopping criteria is met
        if epochs_since_best >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    # Plot metrics after training
    plot_metrics(metrics)

    return model


def plot_metrics(metrics):
    """Построение графиков метрик: Precision, Recall, F1 и Micro F1."""
    epochs = range(1, len(metrics['precision']) + 1)

    plt.figure(figsize=(12, 8))

    plt.plot(epochs, metrics['precision'], label='Точность (Precision, Macro)')
    plt.plot(epochs, metrics['recall'], label='Полнота (Recall, Macro)')
    plt.plot(epochs, metrics['f1'], label='F1-мера (Macro)')
    plt.plot(epochs, metrics['f1_micro'], label='F1-мера (Micro)')

    plt.xlabel('Эпохи')
    plt.ylabel('Значение')
    plt.title('Метрики на валидационных данных по эпохам')
    plt.legend()
    plt.grid(True)
    plt.show()



@torch.no_grad()
def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    # Set device to GPU or CPU based on availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    model.eval()
    out, _ = model((data.x, data.edge_index))
    acc = accuracy(out[data.test_mask].argmax(dim=1), data.y[data.test_mask])
    return acc
