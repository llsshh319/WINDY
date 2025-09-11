import argparse
import os
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
import imageio


def min_max_norm(x: np.ndarray) -> np.ndarray:
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    if x_max - x_min < 1e-12:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def save_grid(frames: np.ndarray, ncols: int, out_path: str, cmap: str = 'GnBu'):
    # frames: [T, H, W] or [T, H, W, 1]
    T = frames.shape[0]
    ncols = min(ncols, T)
    fig, axes = plt.subplots(1, ncols, figsize=(3.0 * ncols, 3.0))
    if ncols == 1:
        axes = [axes]
    for i in range(ncols):
        f = frames[i]
        if f.ndim == 3 and f.shape[-1] == 1:
            f = f[..., 0]
        axes[i].imshow(f, cmap=cmap)
        axes[i].axis('off')
    fig.tight_layout()
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_gif(frames: np.ndarray, out_path: str, cmap: str = 'GnBu'):
    images = []
    for i in range(frames.shape[0]):
        f = frames[i]
        if f.ndim == 3 and f.shape[-1] == 1:
            f = f[..., 0]
        f = (min_max_norm(f) * 255.0).astype(np.uint8)
        images.append(f)
    if not out_path.endswith('.gif'):
        out_path = out_path + '.gif'
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    imageio.mimsave(out_path, images, duration=0.3)


def main():
    p = argparse.ArgumentParser(description='Simple visualization from saved arrays')
    p.add_argument('--base', type=str, required=True, help='Experiment base dir under work_dirs')
    p.add_argument('--index', type=int, default=0, help='Sample index')
    args = p.parse_args()

    saved_dir = osp.join(args.base, 'saved')
    inputs = np.load(osp.join(saved_dir, 'inputs.npy'))
    trues = np.load(osp.join(saved_dir, 'trues.npy'))
    preds = np.load(osp.join(saved_dir, 'preds.npy'))

    # Expect [B, T, C, H, W]
    b = args.index
    x = inputs[b]
    y = trues[b]
    yhat = preds[b]

    # Collapse channel dim if single-channel
    def collapse(a):
        if a.ndim == 4 and a.shape[1] == 1:
            a = a[:, 0]
        elif a.ndim == 4:
            # average channels for quick viewing
            a = a.mean(axis=1)
        return a

    x = collapse(x)
    y = collapse(y)
    yhat = collapse(yhat)

    # Normalize each set independently for visibility
    x_n = min_max_norm(x)
    y_n = min_max_norm(y)
    yhat_n = min_max_norm(yhat)

    vis_dir = osp.join(args.base, 'vis_simple')
    os.makedirs(vis_dir, exist_ok=True)

    save_grid(x_n, ncols=x_n.shape[0], out_path=osp.join(vis_dir, 'input_grid.png'))
    save_grid(y_n, ncols=y_n.shape[0], out_path=osp.join(vis_dir, 'true_grid.png'))
    save_grid(yhat_n, ncols=yhat_n.shape[0], out_path=osp.join(vis_dir, 'pred_grid.png'))

    save_gif(x_n, osp.join(vis_dir, 'input'))
    save_gif(y_n, osp.join(vis_dir, 'true'))
    save_gif(yhat_n, osp.join(vis_dir, 'pred'))

    print('Saved:', vis_dir, sorted(os.listdir(vis_dir)))


if __name__ == '__main__':
    main()


