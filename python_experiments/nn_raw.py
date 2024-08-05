""" Functions very simple, wireframe to port it to C."""


import numpy as np

N_INPUT = 784
N_HIDDEN_1 = 128
# N_HIDDEN_2 = 20
N_OUT = 10
LR = 0.001


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


def outer(_a: np.ndarray, _b: np.ndarray) -> np.ndarray:
	outer_prod = np.zeros((_a.shape[0], _b.shape[0]))
	for i in range(_a.shape[0]):
		for j in range(_b.shape[0]):
			outer_prod[i,j] = _a[i] * _b[j]
	return outer_prod			


def dot_prod(_a: np.ndarray, _b: np.ndarray) -> np.ndarray:
	dot_prod = np.zeros(_b.shape[1])
	for i in range(_a.shape[0]):
		for j in range(_b.shape[1]):
			dot_prod[j] += _a[i] * _b[i,j]
	return dot_prod


def relu(x: float) -> float:
    return float(max(0, x))

            
def hidden_layer(_input: np.ndarray, weights_mat: np.ndarray, bias: np.ndarray) -> np.ndarray:
	z = dot_prod(_input, weights_mat)
	z += bias
	return z


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


def backward_out(logits: np.ndarray, label: np.ndarray, activations: np.ndarray) -> tuple[np.ndarray]:
    dz = logits - label
    dw = outer(activations, dz)
    db = dz
    return dw, db
    

def backward_hidden(inputs: np.ndarray, weights_next: np.ndarray, dz_next: np.ndarray) -> tuple[np.ndarray]:
    da = dot_prod(dz_next, np.transpose(weights_next))
    dz = np.zeros_like(da)
    for i in range(dz.shape[0]):
        if da[i] > 0:
            dz[i] = da[i]
    dw = outer(inputs, dz)
    db = dz
    return dw, db


def update(params: np.ndarray, grads: np.ndarray, lr: float) -> np.ndarray:
    try:
        for i in range(params.shape[0]):
            for j in range(params.shape[1]):
                params[i,j] = params[i,j] - lr * grads[i,j]
    except IndexError:
        for i in range(params.shape[0]):
            params[i] = params[i] - lr * grads[i]
    return params
