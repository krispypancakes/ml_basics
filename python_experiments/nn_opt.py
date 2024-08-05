""" Slightly optimized for keeping it in Python: vectorized operations."""

import numpy as np

N_INPUT = 784
N_HIDDEN_1 = 128
# N_HIDDEN_2 = 20
N_OUT = 10
LR = 0.001


def flatten(_imgs: np.ndarray) -> np.ndarray:
    flattened = [img.flatten() for img in _imgs]
    return np.asarray(flattened)


def relu(x: float) -> float:
    return np.maximum(0, x)
            

def hidden_layer(_input: np.ndarray, weights_mat: np.ndarray, bias: np.ndarray) -> np.ndarray:
	z = _input @ weights_mat
	z += bias
	return z


def onehot(label):
    vec = np.zeros(N_OUT)
    vec[label] = 1
    return vec


def softmax(preds: np.ndarray) -> np.ndarray:
    max_pred = max(preds)
    exp_preds = np.exp(preds - max_pred)
    return exp_preds / exp_preds.sum(axis=0)


def cross_entropy_loss(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    eps = 1e-15
    pred = np.clip(pred, eps, 1-eps) 
    L = -np.sum(target * np.log(pred))
    return L


def backward_out(logits: np.ndarray, label: np.ndarray, activations: np.ndarray) -> tuple[np.ndarray]:
    dz = logits - label
    dw = np.outer(activations, dz)
    db = dz
    return dw, db
    

def backward_hidden(inputs: np.ndarray, weights_next: np.ndarray, dz_next: np.ndarray) -> tuple[np.ndarray]:
    da = dz_next @ np.transpose(weights_next)
    dz = np.where(da > 0, da, 0)
    dw = np.outer(inputs, dz)
    db = dz
    return dw, db


def update(params: np.ndarray, grads: np.ndarray, lr: float) -> np.ndarray:
    return params - lr * grads