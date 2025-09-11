import torch
from torch import nn

from openstl.modules import (ConvSC, ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST,
                             HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock,
                             SwinSubBlock, UniformerSubBlock, VANSubBlock, ViTSubBlock, TAUSubBlock)
from openstl.modules.inr import LIIF, make_coord_cell
from openstl.modules.latent_ode import LatentNeuralODE


class SimVP_Model(nn.Module):
    r"""SimVP Model

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, **kwargs):
        super(SimVP_Model, self).__init__()
        self.args = kwargs
        T, C, H, W = in_shape  # T is pre_seq_length
        self.init_T = T
        # allow overriding core depths/widths via kwargs (from argparse)
        hid_S = int(kwargs.get('hid_S', hid_S))
        hid_T = int(kwargs.get('hid_T', hid_T))
        N_S = int(kwargs.get('N_S', N_S))
        N_T = int(kwargs.get('N_T', N_T))
        mlp_ratio = float(kwargs.get('mlp_ratio', mlp_ratio))
        spatio_kernel_enc = int(kwargs.get('spatio_kernel_enc', spatio_kernel_enc))
        spatio_kernel_dec = int(kwargs.get('spatio_kernel_dec', spatio_kernel_dec))
        # respect act_inplace given by caller
        act_inplace = bool(kwargs.get('act_inplace', act_inplace))
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        # reduce encoder output channels to a small bottleneck (e.g., 4)
        self.enc_out_dim = int(kwargs.get('enc_out_dim', 4))
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace, C_out_last=self.enc_out_dim)
        use_inr_dec = bool(kwargs.get('use_inr_decoder', False))
        if use_inr_dec:
            # Replace conv decoder with INR (LIIF) upsampler to original resolution
            self.dec = None
            self.inr_dec = LIIF(
                in_dim=self.enc_out_dim,
                out_dim=C,
                hidden_list=kwargs.get('inr_hidden', [256, 256, 256]),
                posenc_num_freqs=int(kwargs.get('posenc_num_freqs', 0)),
                posenc_include_input=bool(kwargs.get('posenc_include_input', False)),
                coord_encoding=str(kwargs.get('coord_encoding', 'none')),
                encode_rel_coords=bool(kwargs.get('encode_rel_coords', True)),
                hash_num_levels=int(kwargs.get('hash_num_levels', 12)),
                hash_features_per_level=int(kwargs.get('hash_features_per_level', 2)),
                hash_log2_hashmap_size=int(kwargs.get('hash_log2_hashmap_size', 15)),
                hash_base_resolution=int(kwargs.get('hash_base_resolution', 16)),
                hash_per_level_scale=float(kwargs.get('hash_per_level_scale', 2.0))
            )
        else:
            self.dec = Decoder(self.enc_out_dim, C, N_S, spatio_kernel_dec, act_inplace=act_inplace, C_hid_skip=hid_S)
            self.inr_dec = None

        model_type = 'gsta' if model_type is None else model_type.lower()
        if model_type == 'incepu':
            self.hid = MidIncepNet(T*self.enc_out_dim, hid_T, N_T)
        else:
            self.hid = MidMetaNet(T*self.enc_out_dim, hid_T, N_T,
                input_resolution=(H, W), model_type=model_type,
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)

        # Optional latent Neural ODE dynamics to roll future steps
        self.use_neuralode = bool(kwargs.get('use_neuralode', False))
        if self.use_neuralode:
            self.latent_ode = LatentNeuralODE(in_channels=self.enc_out_dim,
                                              hidden_channels=int(kwargs.get('ode_hidden', 64)),
                                              num_layers=int(kwargs.get('ode_layers', 2)),
                                              method=str(kwargs.get('ode_method', 'rk4')),
                                              step_size=float(kwargs.get('ode_dt', 1.0)))

        # VAE components (optional)
        self.vae = bool(kwargs.get('vae', False))
        if self.vae:
            # Map reduced latent feature maps (enc_out_dim) to mean and log-variance
            self.mu_head = nn.Conv2d(self.enc_out_dim, self.enc_out_dim, kernel_size=1)
            self.logvar_head = nn.Conv2d(self.enc_out_dim, self.enc_out_dim, kernel_size=1)

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape
        # use encoder's reduced channels as z
        C_small = embed.shape[1]
        z = embed.view(B, T, C_small, H_, W_)
        # Align runtime T with construction-time T to avoid channel mismatches
        if T != self.init_T:
            if T < self.init_T:
                # pad by repeating the last frame
                pad_n = self.init_T - T
                pad_tail = z[:, -1:].repeat(1, pad_n, 1, 1, 1)
                z = torch.cat([z, pad_tail], dim=1)
            else:
                # truncate extra frames
                z = z[:, :self.init_T]
            T = self.init_T
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_small, H_, W_)

        kl = None
        if getattr(self, 'vae', False) and (self.args.get('return_kl', False) or self.args.get('vae', False)):
            # Compute mean and log-variance, then reparameterize
            # Heads are constructed in __init__ and move with model.to(device)
            mu = self.mu_head(hid)
            logvar = self.logvar_head(hid)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            hid_sampled = mu + eps * std
            # KL divergence per sample, averaged over batch*time
            kl_map = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            # sum over channel and spatial dims, mean over batch*time
            kl = kl_map.view(B*T, -1).sum(dim=1).mean()
            dec_in = hid_sampled    
        else:
            dec_in = hid

        if self.use_neuralode:
            # Strict: use aft_seq_length (raise KeyError if missing)
            pred_steps = int(self.args['aft_seq_length'])
            # Predict next pred_steps frames given the last latent as initial condition
            z_btchw = dec_in.view(B, T, C_small, H_, W_)
            z0 = z_btchw[:, -1]
            z_future = self.latent_ode(z0, steps=pred_steps)  # (B, pred_steps, C_small, H_, W_)
            dec_seq = z_future.reshape(B*pred_steps, C_small, H_, W_)
            T_out = pred_steps
        else:
            dec_seq = dec_in
            T_out = T

        if self.inr_dec is None:
            Y = self.dec(dec_seq, skip)
            Y = Y.reshape(B, T_out, C, H, W)
        else:
            Hp, Wp = H_, W_
            out = self.inr_dec(
                dec_seq,
                output_size=(H, W),
                bsize=self.args.get('inr_chunk', None),
                train_mask=float(self.args['train_mask'])
            )
            if isinstance(out, dict):
                # out: {'pred': (N,C,H,W), 'mask': (N,1,H,W)} with N=B*T_out
                out['B'] = B
                out['T_out'] = T_out
                if kl is None:
                    return out
                else:
                    return out, kl
            Y = out.view(B, T_out, C, H, W)
        if kl is None:
            return Y
        else:
            return Y, kl


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True, C_out_last=None):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        layers = [ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                         act_inplace=act_inplace)]
        for s in samplings[1:-1]:
            layers.append(ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                                 act_inplace=act_inplace))
        last_out = C_hid if (C_out_last is None) else C_out_last
        layers.append(ConvSC(C_hid, last_out, spatio_kernel, downsampling=samplings[-1],
                             act_inplace=act_inplace))
        self.enc = nn.Sequential(*layers)

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid_small, C_out, N_S, spatio_kernel, act_inplace=True, C_hid_skip=None):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.proj_skip = None
        if C_hid_skip is not None and C_hid_skip != C_hid_small:
            self.proj_skip = nn.Conv2d(C_hid_skip, C_hid_small, kernel_size=1, stride=1, padding=0)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid_small, C_hid_small, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid_small, C_hid_small, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid_small, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        if enc1 is not None and self.proj_skip is not None:
            enc1 = self.proj_skip(enc1)
        Y = self.dec[-1](hid + enc1) if enc1 is not None else self.dec[-1](hid)
        Y = self.readout(Y)
        return Y


class MidIncepNet(nn.Module):
    """The hidden Translator of IncepNet for SimVPv1"""

    def __init__(self, channel_in, channel_hid, N2, incep_ker=[3,5,7,11], groups=8, **kwargs):
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        enc_layers = [gInception_ST(
            channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1,N2-1):
            enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers = [
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups)]
        for i in range(1,N2-1):
            dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_in,
                              incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2-1:
                skips.append(z)
        # decoder
        z = self.dec[0](z)
        for i in range(1,self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1) )

        y = z.reshape(B, T, C, H, W)
        return y


class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'

        if model_type == 'gsta':
            self.block = GASubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'convmixer':
            self.block = ConvMixerSubBlock(in_channels, kernel_size=11, activation=nn.GELU)
        elif model_type == 'convnext':
            self.block = ConvNeXtSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'hornet':
            self.block = HorNetSubBlock(in_channels, mlp_ratio=mlp_ratio, drop_path=drop_path)
        elif model_type in ['mlp', 'mlpmixer']:
            self.block = MLPMixerSubBlock(
                in_channels, input_resolution, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type in ['moga', 'moganet']:
            self.block = MogaSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop_rate=drop, drop_path_rate=drop_path)
        elif model_type == 'poolformer':
            self.block = PoolFormerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'swin':
            self.block = SwinSubBlock(
                in_channels, input_resolution, layer_i=layer_i, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path)
        elif model_type == 'uniformer':
            block_type = 'MHSA' if in_channels == out_channels and layer_i > 0 else 'Conv'
            self.block = UniformerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path, block_type=block_type)
        elif model_type == 'van':
            self.block = VANSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'vit':
            self.block = ViTSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'tau':
            self.block = TAUSubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        else:
            assert False and "Invalid model_type in SimVP"

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y
