import os
import torch
import numpy as np

from openstl.api import BaseExperiment


def main():
    class Args: pass
    args = Args()
    # override
    args.dataname = 'mmnist'
    args.method = 'simvp'
    args.config_file = '/hub_data1/seunghun/OpenSTL/configs/mmnist/simvp/SimVP_gSTA.py'
    args.ex_name = 'mmnist_simvp_gsta_vae'
    args.test = True

    exp = BaseExperiment(args)
    ckpt_path = os.path.join('work_dirs', args.ex_name, 'checkpoints', 'best.ckpt')
    ckpt = torch.load(ckpt_path, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    exp.method.load_state_dict(ckpt['state_dict'])

    loader = exp.data.test_loader
    bx, by = next(iter(loader))

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    exp.method.to(device)
    bx = bx.to(device)

    with torch.no_grad():
        out = exp.method.model(bx)
        preds = out[0] if isinstance(out, tuple) else out

    save_dir = os.path.join('work_dirs', args.ex_name, 'saved')
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'inputs.npy'), bx.detach().cpu().numpy())
    np.save(os.path.join(save_dir, 'trues.npy'), bx.detach().cpu().numpy())
    np.save(os.path.join(save_dir, 'preds.npy'), preds.detach().cpu().numpy())
    print('Saved to', save_dir, list(os.listdir(save_dir)))


if __name__ == '__main__':
    main()









