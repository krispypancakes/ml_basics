""" Functions very simple, wireframe to port it to C."""


import numpy as np

def flatten(_imgs: np.ndarray) -> np.ndarray:
    images = []
    for img in _imgs:
        flat_train = np.zeros(img.shape[1]**2)
        ind = 0
        for c in img: # iterate over channels, although we only have one here
            for i in c: # iterate over rows
                for p in i: # iterate over pixels
                    flat_train[ind] = p
                    ind += 1
        images.append(flat_train)
    return np.asarray(images)


def relu(x: float) -> float:
    return float(max(0, x))


def hidden_layer(_input: np.ndarray, weights_mat: np.ndarray, bias: np.ndarray, last: bool = False) -> tuple[np.ndarray]:
    n_input = weights_mat.shape[0]
    n_output = weights_mat.shape[1]
    z = np.zeros(n_input)
    a = np.zeros(n_output)

    for i in range(n_input):
        for j in range(n_output):
            z[j] += _input[i] * weights_mat[i,j]
        z[j] += bias[j]

    if not last:
        for i in range(n_output):
            a[i] = relu(z[i])
    else:
        a = softmax(z)

    return z, a


def onehot(label):
    vec = np.zeros(N_OUT)
    vec[label] = 1
    return vec


def softmax(preds: np.ndarray) -> np.ndarray:
    logits = np.zeros_like(preds)
    exp_preds = np.zeros_like(preds)
    sum_exp = 0
    max_pred = max(preds)

    for i in range(len(preds)):
        exp_preds[i] = np.exp(preds[i] - max_pred)
        sum_exp += exp_preds[i]

    for i in range(len(preds)):
        logits[i] = exp_preds[i] / sum_exp

    return logits


def cross_entropy_loss(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    eps = 1e-15
    L = 0
    for i in range(N_OUT):
        L += target[i] * np.log(pred[i] + eps)
    return -L

def backward_out(logits: np.ndarray, preds: np.ndarray, labels: np.ndarray) -> tuple[float]:
    da = preds - labels
    dz = da
    dw = np.transpose(logits) * dz
    db = dz
    return dw, db
    

def backward_hidden(inputs: np.ndarray, weights_prev: np.ndarray, dz_prev: np.ndarray) -> np.ndarray:
    da = dz_prev * np.transpose(weights_prev)
    dz = da
    dw = np.transpose(inputs) * dz
    db = dz
    return dw, db


def update(param: float, grad: float, lr: float) -> float:
    return param - lr * grad

