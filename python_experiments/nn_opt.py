""" Slightly optimized for keeping it in Python: vectorized operations."""

import numpy as np


def flatten(_imgs: np.ndarray) -> np.ndarray:
    flattened = [img.flatten() for img in _imgs]
    return np.asarray(flattened)


def relu(x: float) -> float:
    return np.maximum(0, x)
            

def hidden_layer(_input: np.ndarray, weights_mat: np.ndarray, bias: np.ndarray) -> np.ndarray:
	z = _input @ weights_mat
	z += bias
	return z


def onehot(labels: np.ndarray, n_classes: int = 10):
    onehot = np.zeros((labels.size, n_classes))
    onehot[np.arange(labels.size), labels] = 1
    return onehot

def softmax(preds: np.ndarray) -> np.ndarray:
    max_pred = np.max(preds, axis=1, keepdims=True)
    exp_preds = np.exp(preds - max_pred)
    return exp_preds / np.sum(exp_preds, axis=1, keepdims=True)


def cross_entropy_loss(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    eps = 1e-15
    pred = np.clip(pred, eps, 1-eps) 
    L = -np.sum(target * np.log(pred), axis=1)
    return np.mean(L)


def backward_out(logits: np.ndarray, labels: np.ndarray, activations: np.ndarray) -> tuple[np.ndarray]:
    dz = logits - labels
    dw = activations.T @ dz / labels.shape[0]
    db = np.mean(dz, axis=0)
    return dw, db, dz
    

def backward_hidden(inputs: np.ndarray, weights_next: np.ndarray, dz_next: np.ndarray) -> tuple[np.ndarray]:
    da = dz_next @ weights_next.T
    dz = np.where(da>0, da, 0)
    dw = inputs.T @ dz / inputs.shape[0]
    db = np.mean(dz, axis=0)
    return dw, db


def update(params: np.ndarray, grads: np.ndarray, lr: float) -> np.ndarray:
    return params - lr * grads
