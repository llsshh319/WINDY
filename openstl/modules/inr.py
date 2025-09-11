import torch
import torch.nn as nn
import torch.nn.functional as F


def make_coord(shape, flatten=True):
    """Create normalized [-1,1] 2D coordinates for a feature map of shape (H, W).
    Returns tensor of shape (H, W, 2) or (H*W, 2) if flatten.
    """
    # shape can be int (square), tuple/list (H,W), or torch.Size
    if isinstance(shape, int):
        h = w = int(shape)
    else:
        h = int(shape[0])
        w = int(shape[1])
    xs = torch.linspace(-1 + 1e-6, 1 - 1e-6, steps=int(w))
    ys = torch.linspace(-1 + 1e-6, 1 - 1e-6, steps=int(h))
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    coord = torch.stack([grid_y, grid_x], dim=-1)
    if flatten:
        coord = coord.view(-1, 2)
    return coord


def make_coord_cell(batch_size, h, w, device):
    """Return (coord, cell) batched for LIIF. coord, cell shapes: (B, H*W, 2)."""
    # normalize types
    if isinstance(h, (tuple, list)):
        h = h[0]
    if isinstance(w, (tuple, list)):
        w = w[0]
    coord = make_coord((int(h), int(w)), flatten=True).to(device)
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    coord = coord.unsqueeze(0).repeat(batch_size, 1, 1)
    cell = cell.unsqueeze(0).repeat(batch_size, 1, 1)
    return coord, cell


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_list:
            layers += [nn.Linear(last, h), nn.GELU()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fourier_encode(x, num_bands: int, include_input: bool = False):
    """Apply NeRF-style Fourier positional encoding to coordinates in [-1,1].
    x: (B, Q, D) where D is typically 2 for (y,x) here.
    Returns: (B, Q, D * 2 * num_bands [+ D if include_input])
    Frequencies: 2^k * pi for k in [0, num_bands-1].
    """
    if num_bands <= 0:
        return x if include_input else x.new_zeros(x.shape[:-1] + (0,))
    device = x.device
    D = x.shape[-1]
    k = torch.arange(num_bands, device=device, dtype=x.dtype)
    freqs = (2.0 ** k) * torch.pi  # (num_bands,)
    # Expand: (B,Q,D) * (num_bands,) -> (B,Q,D,num_bands)
    xb = x.unsqueeze(-1) * freqs  # broadcast multiply
    sin = torch.sin(xb)
    cos = torch.cos(xb)
    # concat on last dim -> (B,Q,D,2*num_bands) then flatten D dimension
    enc = torch.cat([sin, cos], dim=-1).reshape(x.shape[0], x.shape[1], D * 2 * num_bands)
    if include_input:
        enc = torch.cat([x, enc], dim=-1)
    return enc


class HashGridEncoder2D(nn.Module):
    """Simplified 2D multi-resolution hash grid encoder (Instant-NGP style).
    Inputs are expected in [-1, 1]. Internally mapped to [0, 1].
    """
    def __init__(self, num_levels: int = 12, features_per_level: int = 2,
                 log2_hashmap_size: int = 15, base_resolution: int = 16,
                 per_level_scale: float = 2.0):
        super().__init__()
        self.num_levels = int(num_levels)
        self.features_per_level = int(features_per_level)
        self.hashmap_size = 1 << int(log2_hashmap_size)
        self.base_resolution = int(base_resolution)
        self.per_level_scale = float(per_level_scale)

        self.embeddings = nn.ParameterList([
            nn.Parameter(torch.empty(self.hashmap_size, self.features_per_level))
            for _ in range(self.num_levels)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for table in self.embeddings:
            nn.init.uniform_(table, a=-1e-4, b=1e-4)

    @staticmethod
    def fast_hash(ix: torch.Tensor, iy: torch.Tensor, size: int) -> torch.Tensor:
        # Spatial hash with 32-bit primes to avoid Python long overflows
        ix_i = ix.to(torch.int64)
        iy_i = iy.to(torch.int64)
        p1 = torch.tensor(73856093, dtype=torch.int64, device=ix_i.device)
        p2 = torch.tensor(19349663, dtype=torch.int64, device=ix_i.device)
        h = (ix_i * p1) ^ (iy_i * p2)
        # Map to [0, size)
        if (size & (size - 1)) == 0:
            mask = torch.tensor(size - 1, dtype=torch.int64, device=ix_i.device)
            return (h & mask).to(torch.long)
        size_t = torch.tensor(size, dtype=torch.int64, device=ix_i.device)
        return torch.remainder(h.abs(), size_t).to(torch.long)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode coords.
        coords: (B, Q, 2) in [-1, 1]
        returns: (B, Q, num_levels * features_per_level)
        """
        B, Q, D = coords.shape
        assert D == 2
        # map to [0,1)
        x = (coords + 1.0) * 0.5
        feats = []
        device = coords.device
        for lvl in range(self.num_levels):
            res = int(self.base_resolution * (self.per_level_scale ** lvl))
            # Continuous grid coords in [0, res)
            gx = x[..., 1] * (res - 1)
            gy = x[..., 0] * (res - 1)
            ix0 = torch.floor(gx).clamp(0, res - 1)
            iy0 = torch.floor(gy).clamp(0, res - 1)
            fx = (gx - ix0).to(coords.dtype)
            fy = (gy - iy0).to(coords.dtype)
            ix0 = ix0.long()
            iy0 = iy0.long()
            ix1 = (ix0 + 1)
            iy1 = (iy0 + 1)

            # Hash 4 corners
            h00 = self.fast_hash(ix0, iy0, self.hashmap_size)
            h10 = self.fast_hash(ix1, iy0, self.hashmap_size)
            h01 = self.fast_hash(ix0, iy1, self.hashmap_size)
            h11 = self.fast_hash(ix1, iy1, self.hashmap_size)

            table = self.embeddings[lvl]
            e00 = table[h00]
            e10 = table[h10]
            e01 = table[h01]
            e11 = table[h11]

            # Bilinear interpolation
            w00 = (1 - fx) * (1 - fy)
            w10 = fx * (1 - fy)
            w01 = (1 - fx) * fy
            w11 = fx * fy
            feat = (
                e00 * w00.unsqueeze(-1)
                + e10 * w10.unsqueeze(-1)
                + e01 * w01.unsqueeze(-1)
                + e11 * w11.unsqueeze(-1)
            )  # (B,Q,F)
            feats.append(feat)

        return torch.cat(feats, dim=-1)


class LIIF(nn.Module):
    """
    Lightweight Implicit Image Function decoder.
    Given a low‑res feature map (B,C,Hp,Wp), predicts values at arbitrary (H,W) coordinates.
    """
    def __init__(self, in_dim, out_dim, hidden_list, cell_decode=False, local_ensemble=True,
                 posenc_num_freqs: int = 0, posenc_include_input: bool = False,
                 coord_encoding: str = 'none', encode_rel_coords: bool = True,
                 hash_num_levels: int = 12, hash_features_per_level: int = 2,
                 hash_log2_hashmap_size: int = 15, hash_base_resolution: int = 16,
                 hash_per_level_scale: float = 2.0):
        super().__init__()
        self.cell_decode = cell_decode
        self.local_ensemble = local_ensemble
        self.coord_encoding = str(coord_encoding).lower()
        self.encode_rel_coords = bool(encode_rel_coords)
        # Fourier params
        self.posenc_num_freqs = int(posenc_num_freqs)
        self.posenc_include_input = bool(posenc_include_input)
        # Hash grid params
        self.hash_num_levels = int(hash_num_levels)
        self.hash_features_per_level = int(hash_features_per_level)
        self.hash_log2_hashmap_size = int(hash_log2_hashmap_size)
        self.hash_base_resolution = int(hash_base_resolution)
        self.hash_per_level_scale = float(hash_per_level_scale)

        def _coord_feat_dim():
            if self.coord_encoding == 'fourier':
                if self.posenc_num_freqs > 0:
                    return (2 * 2 * self.posenc_num_freqs) + (2 if self.posenc_include_input else 0)
                return 0
            if self.coord_encoding == 'hashgrid':
                return self.hash_num_levels * self.hash_features_per_level
            return 0

        coord_feat_dim = _coord_feat_dim() + 2
        mlp_in = in_dim + coord_feat_dim + (2 if cell_decode else 0)
        self.imnet = MLP(mlp_in, out_dim, hidden_list)

        if self.coord_encoding == 'hashgrid':
            self.hash_encoder = HashGridEncoder2D(
                num_levels=self.hash_num_levels,
                features_per_level=self.hash_features_per_level,
                log2_hashmap_size=self.hash_log2_hashmap_size,
                base_resolution=self.hash_base_resolution,
                per_level_scale=self.hash_per_level_scale
            )
        else:
            self.hash_encoder = None

    def _query(self, feat, coord, cell):
        B, C, Hp, Wp = feat.shape
        if self.local_ensemble:
            vx_lst, vy_lst = [-1, 1], [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0.0

        rx = 1.0 / Hp
        ry = 1.0 / Wp

        # precompute feature grid coords in [-1,1]
        feat_coord = make_coord((Hp, Wp), flatten=False).to(feat.device)
        feat_coord = feat_coord.permute(2, 0, 1).unsqueeze(0).expand(B, 2, Hp, Wp)

        preds, areas = [], []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] = (coord_[:, :, 0] + vx * rx + eps_shift).clamp(-1 + 1e-6, 1 - 1e-6)
                coord_[:, :, 1] = (coord_[:, :, 1] + vy * ry + eps_shift).clamp(-1 + 1e-6, 1 - 1e-6)
                # sample nearest features and positions
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= Hp
                rel_coord[:, :, 1] *= Wp
                # Choose coordinates to encode
                enc_src = rel_coord if self.encode_rel_coords else coord
                if self.coord_encoding == 'fourier':
                    if self.posenc_num_freqs > 0:
                        coord_feat = fourier_encode(enc_src, self.posenc_num_freqs, include_input=self.posenc_include_input)
                    else:
                        coord_feat = enc_src
                elif self.coord_encoding == 'hashgrid':
                    coord_feat = self.hash_encoder(coord)
                    coord_feat = torch.cat([coord_feat, enc_src], dim=-1)
                else:
                    coord_feat = enc_src
                inp = torch.cat([q_feat, coord_feat], dim=-1)
                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= Hp
                    rel_cell[:, :, 1] *= Wp
                    inp = torch.cat([inp, rel_cell], dim=-1)
                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)
                area = (rel_coord[:, :, 0].abs() * rel_coord[:, :, 1].abs()) + 1e-9
                areas.append(area)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            areas[0], areas[3] = areas[3], areas[0]
            areas[1], areas[2] = areas[2], areas[1]
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, feat, output_size, train_mask, coord=None, cell=None, return_img=True, bsize=None):
        B, C, Hp, Wp = feat.shape
        if isinstance(output_size, int):
            H = W = int(output_size)
        else:
            H, W = int(output_size[0]), int(output_size[1])
        if coord is None or cell is None:
            coord, cell = make_coord_cell(B, H, W, feat.device)
        # Randomly subsample coordinates during training to save memory
        masked = False
        if self.training and (train_mask is not None) and (float(train_mask) < 1.0):
            Q = coord.shape[1]
            k = max(1, int(Q * float(train_mask)))
            # Same random subset for all items in batch for simplicity
            idx = torch.randperm(Q, device=feat.device)[:k]
            mask = torch.zeros(H * W, dtype=torch.bool, device=feat.device)
            mask[idx] = True
            coord = coord[:, mask, :]
            cell = cell[:, mask, :]
            masked = True
        # If training and chunk size not provided, default to no chunking; otherwise honor given bsize
        chunk = 0 if (self.training and (bsize is None)) else (int(bsize) if bsize is not None else 65536)
        if chunk > 0:
            n = coord.shape[1]
            preds, ql = [], 0
            while ql < n:
                qr = min(ql + chunk, n)
                preds.append(self._query(feat, coord[:, ql:qr, :], cell[:, ql:qr, :]))
                ql = qr
            out = torch.cat(preds, dim=1)
        else:
            out = self._query(feat, coord, cell)
        # If masked sampling was used, scatter back to dense map and return mask
        if masked:
            # out: (N, Q, C); broadcast a 1D boolean mask over batch to select target pixels
            N, Q_sel, C_out = out.shape
            return {
                'pred': out,    # (N, Q, C)
                'mask': mask,   # (H*W,) bool; applies to dim=1 of flattened target
            }
        if return_img:
            out = out.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return out


