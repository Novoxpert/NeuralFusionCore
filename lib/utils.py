
import os, numpy as np, matplotlib.pyplot as plt
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def plot_equity(dates, equity, out_png=None):
    plt.figure(figsize=(10,5)); plt.plot(dates, equity, label='Strategy')
    plt.legend(); plt.grid(True); plt.xticks(rotation=45)
    plt.title('Portfolio Cumulative Return'); plt.tight_layout()
    if out_png:
        ensure_dir(os.path.dirname(out_png)); plt.savefig(out_png, dpi=140)
    plt.show()

def atomic_model_swap(src_path, dest_path):
    """
    Atomically replace dest_path with src_path (works locally). You might want to do
    object storage / tagging for distributed envs. This is a simple approach.
    """
    import os, shutil
    bak = dest_path + ".bak"
    if os.path.exists(dest_path):
        shutil.move(dest_path, bak)
    shutil.move(src_path, dest_path)
    if os.path.exists(bak):
        os.remove(bak)