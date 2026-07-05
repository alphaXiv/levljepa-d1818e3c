import torch
import torch.distributed as dist


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer.

    Regularizes a marginal embedding distribution toward an isotropic Gaussian
    via an Epps-Pulley goodness-of-fit statistic on random 1-D projections.

    With ``distributed=True`` the empirical characteristic function is averaged
    across all ranks before the statistic is formed, so the test sees the
    global batch rather than the per-rank batch. The statistic is scaled by the
    *local* batch in both modes, so ``lambda`` is comparable regardless of
    ``distributed`` -- enabling it only swaps the noisy per-rank estimate of the
    characteristic function for a less-noisy global one, it does not change the
    loss magnitude.
    """

    def __init__(self, knots=17, num_proj=4096, distributed=False):
        super().__init__()
        self.num_proj = num_proj
        self.distributed = distributed
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """proj: ``(..., B, D)`` -- the batch dimension is dim ``-2``."""
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))

        # empirical characteristic function (cos/sin means over the batch dim)
        x_t = (proj @ A).unsqueeze(-1) * self.t

        if self.distributed and dist.is_available() and dist.is_initialized():
            # Average the characteristic function across ranks (global estimate),
            # but scale the statistic by the LOCAL batch -- same magnitude as the
            # non-distributed path, so lambda is comparable across modes.
            ecf = torch.stack((x_t.cos().mean(-3), x_t.sin().mean(-3)))
            dist.all_reduce(ecf, op=dist.ReduceOp.AVG)
            err = (ecf[0] - self.phi).square() + ecf[1].square()
        else:
            err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()

        # epps-pulley statistic (scaled by local batch in both modes)
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
