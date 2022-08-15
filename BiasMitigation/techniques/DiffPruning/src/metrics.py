import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def f1score(predictions: torch.Tensor, labels: torch.Tensor, **kwargs) -> float:
    pred_np = predictions.numpy()
    labels_np = labels.numpy()
    return f1_score(labels_np, pred_np, **kwargs)


def accuracy(predictions: torch.Tensor, labels: torch.Tensor, balanced: bool = False) -> float:
    pred_np = predictions.numpy()
    labels_np = labels.numpy()
    if balanced:
        return balanced_accuracy_score(labels_np, pred_np)
    return accuracy_score(labels_np, pred_np)
