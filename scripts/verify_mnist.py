"""
MNIST result verification — inference only.

Loads a saved checkpoint and reproduces the reported test accuracy.
No training code is included.
"""
import os, pickle
import numpy as np


def load_mnist():
    npz = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist.npz')
    d = np.load(npz)
    X = (d['X'] > 128).astype(np.float64)
    y = d['y']
    return X[60000:], y[60000:]


def forward(W1, W2, X):
    """Forward pass: input -> gate (ReLU) -> output."""
    H = np.maximum(X @ W1, 0)
    return np.nan_to_num(H @ W2, nan=0)


def verify(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f'Checkpoint not found: {ckpt_path}')
        return

    with open(ckpt_path, 'rb') as f:
        state = pickle.load(f)

    W1, W2 = state['W1'], state['W2']
    epoch = state['epoch']
    print(f'Loaded checkpoint: epoch {epoch}')
    print(f'  W1: {W1.shape}, W2: {W2.shape}')

    X_te, y_te = load_mnist()

    # full test set
    O = forward(W1, W2, X_te)
    preds = O.argmax(axis=1)
    acc = (preds == y_te).mean()
    print(f'  Test accuracy (n={len(y_te)}): {acc:.4f}')

    # per-digit accuracy
    print(f'\n  Per-digit accuracy:')
    for d in range(10):
        mask = y_te == d
        if mask.sum() == 0:
            continue
        digit_acc = (preds[mask] == d).mean()
        print(f'    digit {d}: {digit_acc:.3f} (n={mask.sum()})')


if __name__ == '__main__':
    ckpt = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist_mat_ckpt.pkl')
    verify(ckpt)
