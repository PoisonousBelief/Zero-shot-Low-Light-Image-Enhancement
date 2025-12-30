# put this in a file like physical_consistency_losses.py
import torch
import torch.nn.functional as F
import math

# ---------- helpers ----------
def make_gaussian_kernel(radius, sigma, device, dtype=torch.float32):
    """Return 1D gaussian kernel tensor of shape (k,) on correct device/dtype."""
    size = int(radius) * 2 + 1
    coords = torch.arange(size, dtype=dtype, device=device) - float(radius)
    kernel1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel1d = kernel1d / kernel1d.sum()
    return kernel1d  # shape (k,)

def gaussian_blur_tensor(x, radius=15, sigma=3.0):
    """
    Separable gaussian blur using group conv (fast and correct).
    x: (B, C, H, W)
    radius: int radius for kernel size = 2*radius+1
    sigma: gaussian sigma
    returns: blurred tensor same shape as x
    """
    if x.dim() != 4:
        raise ValueError("gaussian_blur_tensor expects input shape (B,C,H,W)")

    C = x.shape[1]
    device = x.device
    dtype = x.dtype

    # create 1D kernel
    k1d = make_gaussian_kernel(radius, sigma, device=device, dtype=dtype)  # (k,)
    k_size = k1d.shape[0]

    # horizontal kernel: shape (C,1,1,k)
    k_h = k1d.view(1, 1, 1, k_size).repeat(C, 1, 1, 1)
    # vertical kernel: shape (C,1,k,1)
    k_v = k1d.view(1, 1, k_size, 1).repeat(C, 1, 1, 1)

    pad_h = (radius, radius, 0, 0)  # (left, right, top, bottom)
    pad_v = (0, 0, radius, radius)

    # horizontal convolution
    x_pad = F.pad(x, pad_h, mode='reflect')
    x_h = F.conv2d(x_pad, weight=k_h, bias=None, stride=1, padding=0, groups=C)

    # vertical convolution
    x_pad2 = F.pad(x_h, pad_v, mode='reflect')
    out = F.conv2d(x_pad2, weight=k_v, bias=None, stride=1, padding=0, groups=C)

    return out

def rgb_to_gray(x):
    # x: (B,3,H,W)
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b

def sobel_edge_map_gray(y):
    # y: (B,1,H,W)
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=y.dtype, device=y.device).view(1,1,3,3)
    sobel_y = sobel_x.transpose(2,3)
    gx = F.conv2d(F.pad(y, (1,1,1,1), mode='reflect'), sobel_x)
    gy = F.conv2d(F.pad(y, (1,1,1,1), mode='reflect'), sobel_y)
    mag = torch.sqrt(gx*gx + gy*gy + 1e-12)
    return mag

def total_variation(x):
    # x: (B,1,H,W) or (B,C,H,W)
    dh = torch.abs(x[..., 1:, :] - x[..., :-1, :]).mean()
    dw = torch.abs(x[..., :, 1:] - x[..., :, :-1]).mean()
    return dh + dw

# ---------- main scoring function ----------
def physical_consistency_losses(orig_rgb, enh_rgb,
                                params=None):
    """
    orig_rgb, enh_rgb: torch.Tensor in [0,1], shape (B,3,H,W) or (3,H,W)
    returns dict of losses (per-batch average scalars)
    """
    if params is None:
        params = {}
    # defaults
    #target_lum = params.get('target_lum', 0.45)   # desired mean luminance after enhancement
    target_lum = params.get('target_lum', 0.3)   # desired mean luminance after enhancement
    blur_radius = params.get('blur_radius', 15)
    blur_sigma = params.get('blur_sigma', 3.0)
    blur_radius_hf = params.get('blur_radius_hf', 3)
    blur_sigma_hf = params.get('blur_sigma_hf', 1.0)
    alpha_noise = params.get('alpha_noise', 1.2)
    eps = 1e-6
    device = orig_rgb.device

    # ensure batch dim
    single_input = False
    if orig_rgb.dim() == 3:
        orig_rgb = orig_rgb.unsqueeze(0); enh_rgb = enh_rgb.unsqueeze(0); single_input = True

    B = orig_rgb.shape[0]

    # clamp
    orig = orig_rgb.clamp(0.0, 1.0)
    enh = enh_rgb.clamp(0.0, 1.0)

    # luminance
    Y_orig = rgb_to_gray(orig)      # (B,1,H,W)
    Y_enh = rgb_to_gray(enh)

    # illumination estimate (Gaussian blur of luminance)
    L_orig = gaussian_blur_tensor(Y_orig, radius=blur_radius, sigma=blur_sigma)
    L_enh = gaussian_blur_tensor(Y_enh, radius=blur_radius, sigma=blur_sigma)

    # reflectance estimates
    R_orig = Y_orig / (L_orig + eps)
    R_enh = Y_enh / (L_enh + eps)

    # exposure loss (mean luminance to target)
    mean_lum_enh = Y_enh.mean(dim=[1,2,3])
    L_exp = ((mean_lum_enh - target_lum) ** 2).mean()

    # reflectance consistency (L1)
    L_refl = F.l1_loss(R_enh, R_orig, reduction='mean')

    # illumination smoothness (TV on illumination); we expect smooth illumination
    L_illum_tv = total_variation(L_enh)

    # edge consistency (Sobel on luminance)
    E_orig = sobel_edge_map_gray(Y_orig)
    E_enh = sobel_edge_map_gray(Y_enh)
    L_edge = F.l1_loss(E_enh, E_orig, reduction='mean')

    # noise / high-frequency constraint
    HF_orig = Y_orig - gaussian_blur_tensor(Y_orig, radius=blur_radius_hf, sigma=blur_sigma_hf)
    HF_enh = Y_enh - gaussian_blur_tensor(Y_enh, radius=blur_radius_hf, sigma=blur_sigma_hf)
    hf_orig_norm = HF_orig.abs().mean(dim=[1,2,3]) + eps
    hf_enh_norm = HF_enh.abs().mean(dim=[1,2,3])
    rel = hf_enh_norm - alpha_noise * hf_orig_norm  # positive -> extra high freq
    rel = torch.relu(rel)
    L_noise = rel.mean()

    # chroma / color preservation (avoid color shifts)
    chroma_orig = (orig[:, :3, :, :] / (Y_orig + eps))
    chroma_enh = (enh[:, :3, :, :] / (Y_enh + eps))
    L_chroma = F.l1_loss(chroma_enh, chroma_orig, reduction='mean')

    # compose dict
    losses = {
        'exp': L_exp,
        'refl': L_refl,
        'illum_tv': L_illum_tv,
        'edge': L_edge,
        'noise': L_noise,
        'chroma': L_chroma
    }
    # default weights (you can tune)
    weights = params.get('weights', {
        'exp': 1.0,
        'refl': 1.0,
        'illum_tv': 0.5,
        'edge': 0.8,
        'noise': 2.0,
        'chroma': 0.6
    })
    total = 0.0
    for k, v in losses.items():
        total = total + weights.get(k, 1.0) * v
    losses['total'] = total

    # if single_input, return scalars
    if single_input:
        return {k: float(v.detach().cpu().item()) for k, v in losses.items()}
    else:
        # per-batch scalars
        return {k: v.detach().cpu().mean().item() for k, v in losses.items()}


# ---------- utility: score a list of candidates ----------
def score_candidates(orig_rgb, list_enh_candidates, params=None):
    """
    orig_rgb: torch.Tensor (3,H,W) or (1,3,H,W) in [0,1]
    list_enh_candidates: list of torch.Tensor with same shape as orig_rgb
    returns list of dicts with the losses for each candidate and returns index of best (min total)
    """
    results = []
    for c in list_enh_candidates:
        losses = physical_consistency_losses(orig_rgb, c, params=params)
        results.append(losses)
    # pick best by min total
    totals = [r['total'] for r in results]
    best_idx = int(torch.argmin(torch.tensor(totals)).item())
    return results, best_idx
