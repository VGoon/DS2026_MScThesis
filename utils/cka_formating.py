import numpy as np
from CKA import linear_CKA

def get_activation_pytorch(name, activations):
    def hook(module, input, output):
        activations.setdefault(name, []).append(output.detach())
    return hook

def reshape_for_cka_pytorch(tensor):
    tensor = tensor.mean(dim=[2, 3])
    return tensor
    
def compute_cka_matrix_pytorch(activations_dict):
    layers = list(activations_dict.keys())
    n = len(layers)
    
    cka_matrix = np.zeros((n, n))

    reshaped = {}
    for k, v in activations_dict.items():
        reshaped[k] = reshape_for_cka_pytorch(v)

    for i in range(n):
        for j in range(n):
            X = reshaped[layers[i]]
            Y = reshaped[layers[j]]

            if X.shape[0] != Y.shape[0]:
                cka_matrix[i, j] = float('nan')  # skip
                continue

            cka_matrix[i, j] = linear_CKA(X, Y)

    return cka_matrix, layers


def reshape_for_cka_tensorflow(tensor):
    # tensor: (B, H, W, C)
    return tensor.mean(axis=(1, 2))  # → (B, C)

def compute_cka_matrix_tensorflow(activations_dict):
    layers = list(activations_dict.keys())
    n = len(layers)
    
    cka_matrix = np.zeros((n, n))

    reshaped = {}
    for k, v in activations_dict.items():
        reshaped[k] = reshape_for_cka_tensorflow(v)

    for i in range(n):
        for j in range(n):
            X = reshaped[layers[i]]
            Y = reshaped[layers[j]]

            cka_matrix[i, j] = linear_CKA(X, Y)

    return cka_matrix, layers

def reshape_for_cka(x):
    # works for both PT and TF (after numpy conversion)

    if len(x.shape) == 4:
        # PT: (N, C, H, W)
        # TF: (N, H, W, C)

        # detect format
        if x.shape[1] < 10:  # crude check: channels small → PT
            x = np.transpose(x, (0, 2, 3, 1))  # CHW → HWC

        N = x.shape[0]
        return x.reshape(N, -1)

    elif len(x.shape) == 2:
        return x  # already flat

    else:
        raise ValueError("Unsupported shape")
    
def compute_cka_matrix(activations_dict):
    layers = list(activations_dict.keys())
    n = len(layers)

    cka_matrix = np.zeros((n, n))

    reshaped = {}
    for k, v in activations_dict.items():
        reshaped[k] = reshape_for_cka(v)

    for i in range(n):
        for j in range(n):
            X = reshaped[layers[i]]
            Y = reshaped[layers[j]]

            if X.shape[0] != Y.shape[0]:
                cka_matrix[i, j] = float('nan')
                continue

            cka_matrix[i, j] = linear_CKA(X, Y)

    return cka_matrix, layers

def compute_cross_cka(pt_activations, tf_activations):
    stages = ["stage1", "stage2", "stage3", "stage4"]
    cka_matrix = np.zeros((4, 4))

    for i, p in enumerate(stages):
        for j, t in enumerate(stages):
            X = reshape_for_cka(pt_activations[p])
            Y = reshape_for_cka(tf_activations[t])

            if X.shape[0] != Y.shape[0]:
                cka_matrix[i, j] = np.nan
                continue

            cka_matrix[i, j] = linear_CKA(X, Y)

    return cka_matrix, stages