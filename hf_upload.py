"""Upload CGN MNIST results to Hugging Face."""
from huggingface_hub import HfApi, create_repo
import os

REPO_ID = "alexcynn/cgn-mnist"
LOCAL_DIR = os.path.join(os.path.dirname(__file__))

api = HfApi()

# Create repo (model type)
try:
    create_repo(REPO_ID, repo_type="model", exist_ok=True)
    print(f"Repo ready: {REPO_ID}")
except Exception as e:
    print(f"Repo: {e}")

# Upload files
files = [
    ("data/mnist_mat_ckpt.pkl", "checkpoint/mnist_mat_ckpt.pkl"),
    ("scripts/verify_mnist.py", "scripts/verify_mnist.py"),
    ("scripts/visualize_gates.py", "scripts/visualize_gates.py"),
    ("scripts/compare_resolution.py", "scripts/compare_resolution.py"),
    ("scripts/diagram_cgn.py", "scripts/diagram_cgn.py"),
    ("results/mnist_h128_log.txt", "results/mnist_h128_log.txt"),
    ("results/mnist_prune_log.txt", "results/mnist_prune_log.txt"),
    ("figures/gate_receptive_fields.png", "figures/gate_receptive_fields.png"),
    ("figures/gate_output_votes.png", "figures/gate_output_votes.png"),
    ("figures/cgn_vs_cnn.png", "figures/cgn_vs_cnn.png"),
    ("figures/confluence_gate_detail.png", "figures/confluence_gate_detail.png"),
    ("figures/cgn_vs_cnn_resolution.png", "figures/cgn_vs_cnn_resolution.png"),
]

for local, remote in files:
    local_path = os.path.join(LOCAL_DIR, local)
    if os.path.exists(local_path):
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote,
            repo_id=REPO_ID,
        )
        print(f"  Uploaded: {remote}")
    else:
        print(f"  SKIP (not found): {local}")

print("Done.")
