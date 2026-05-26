import torch
import math
import torch.nn.functional as F


def preprocess_single(video, resolution, sequence_length=None):
    # video: THWC, {0, ..., 255}
    video = video.permute(0, 3, 1, 2).float() / 255.  # TCHW
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear', align_corners=False)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous()  # CTHW

    video -= 0.5

    return video


def preprocess(videos, target_resolution=224):
    # videos in {0, ..., 255} as np.uint8 array
    videos = torch.from_numpy(videos)
    videos = torch.stack([preprocess_single(video, target_resolution) for video in videos])
    return videos * 2  # [-0.5, 0.5] -> [-1, 1]


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L161
def _symmetric_matrix_square_root(mat, eps=1e-10):
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L400
def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()

    fact = 1.0 / (m.size(1) - 1)  # unbiased estimate
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


def frechet_distance(x1, x2):
    x1 = x1.flatten(start_dim=1)
    x2 = x2.flatten(start_dim=1)
    m, m_w = x1.mean(dim=0), x2.mean(dim=0)
    sigma, sigma_w = cov(x1, rowvar=False), cov(x2, rowvar=False)

    sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)
    trace = torch.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

    mean = torch.sum((m - m_w) ** 2)
    fd = trace + mean
    return fd


def get_logits(i3d, videos, device):
    with torch.no_grad():
        chunks = torch.split(videos, 16, dim=2)
        logits = []
        for chunk in chunks:
            logits.append(i3d(chunk.to(device)))
        logits = torch.cat(logits, dim=0)
        return logits


def get_fvd_logits(videos, i3d, device):
    videos = preprocess(videos)
    embeddings = get_logits(i3d, videos, device)
    return embeddings


def load_i3d_pretrained(device=torch.device('cpu')):
    from .pytorch_i3d import InceptionI3d
    i3d = InceptionI3d(400, in_channels=3).to(device)
    # TODO: set your custom i3d path. (in this case we use -> `EgoControl/cosmos-predict2/checkpoints/`)
    filepath = 'cosmos-predict2/checkpoints/i3d_pretrained_400.pt'
    i3d.load_state_dict(torch.load(filepath, map_location=device))
    i3d.eval()
    return i3d

def compute_fvd(gt_videos, pred_videos, device):
    """
    Compute FVD between ground truth and predicted videos.
    """
    from torchvision.io import read_video
    import torch

    i3d = load_i3d_pretrained(device='cuda')
    gt = []
    pred = []
    for video in gt_videos:
        try:
            v = read_video(video)[0]
            if v.shape[0] > 1:
                gt.append(v)
        except:
            pass
    for video in pred_videos:
        try:
            v = read_video(video)[0]
            if v.shape[0] > 1:
                pred.append(v)
        except:
            pass

    batch = 16  # Batch size for FVD computation

    gt_fvd_emb = torch.cat([get_fvd_logits(torch.stack(gt[i:i + batch]).numpy(), i3d=i3d, device=device)
                            for i in range(0, len(gt), batch)], dim=0)
    pred_fvd_emb = torch.cat([get_fvd_logits(torch.stack(pred[i:i + batch]).numpy(), i3d=i3d, device=device)
                              for i in range(0, len(pred), batch)], dim=0)

    return frechet_distance(pred_fvd_emb, gt_fvd_emb)