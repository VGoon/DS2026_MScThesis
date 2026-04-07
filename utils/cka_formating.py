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