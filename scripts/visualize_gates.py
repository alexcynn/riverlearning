"""
Gate visualization — renders each hidden node's receptive field
(W1 column) as a 28x28 image.

Shows which pixel combinations open each gate.
"""
import os, pickle
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print('matplotlib not installed. Install with: pip install matplotlib')


def load_ckpt(path):
    with open(path, 'rb') as f:
        state = pickle.load(f)
    return state['W1'], state['W2']


def visualize_gates(W1, W2, out_dir, top_k=32):
    """Visualize receptive fields of top_k gates ranked by W2 influence."""
    n_h = W1.shape[1]

    # gate importance: L1 norm of W2 row (influence on output)
    importance = np.abs(W2).sum(axis=1)
    top_idx = np.argsort(importance)[::-1][:top_k]

    # each gate's preferred output class
    preferred_class = W2.argmax(axis=1)

    # grid plot
    cols = 8
    rows = (top_k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))
    axes = axes.flatten()

    for i, idx in enumerate(top_idx):
        rf = W1[:, idx].reshape(28, 28)
        axes[i].imshow(rf, cmap='RdBu_r', vmin=-rf.max(), vmax=rf.max())
        axes[i].set_title(f'g{idx}→{preferred_class[idx]}', fontsize=8)
        axes[i].axis('off')

    for i in range(len(top_idx), len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Top {top_k} Gate Receptive Fields (W1 columns, 28x28)\n'
                 f'Title: gate_id → preferred_digit', fontsize=10)
    plt.tight_layout()

    path = os.path.join(out_dir, 'gate_receptive_fields.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def visualize_votes(W2, out_dir):
    """Heatmap of each gate's output vote (W2 row)."""
    # top 32 gates
    importance = np.abs(W2).sum(axis=1)
    top_idx = np.argsort(importance)[::-1][:32]

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(W2[top_idx], cmap='RdBu_r', aspect='auto')
    ax.set_xlabel('Output digit')
    ax.set_ylabel('Gate (sorted by importance)')
    ax.set_xticks(range(10))
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([f'g{i}' for i in top_idx], fontsize=7)
    plt.colorbar(im, ax=ax, label='Vote strength')
    plt.title('Gate Output Votes (W2 rows)')
    plt.tight_layout()

    path = os.path.join(out_dir, 'gate_output_votes.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


if __name__ == '__main__':
    if not HAS_PLT:
        raise SystemExit(1)

    ckpt = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist_mat_ckpt.pkl')
    if not os.path.exists(ckpt):
        print(f'Checkpoint not found: {ckpt}')
        raise SystemExit(1)

    W1, W2 = load_ckpt(ckpt)
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    visualize_gates(W1, W2, out_dir)
    visualize_votes(W2, out_dir)
    print('Done.')
