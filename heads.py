"""Slide-level heads over frozen tile embeddings.

Every head is fitted on a list of bags (one float16 array [n_tiles, dim] per slide) and
labels, picks its one hyperparameter on a separate validation list, and scores new bags.
Tile embeddings are standardised with statistics from the fitting bags only.

    mean   logistic regression on the mean of the standardised tile embeddings
    max    logistic regression on the per-dimension maximum
    abmil  gated attention MIL (Ilse, Tomczak and Welling, 2018)
"""
import copy

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import config


def tile_stats(bags: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of every embedding dimension over all tiles of `bags`."""
    total = np.zeros(bags[0].shape[1])
    total_sq = np.zeros(bags[0].shape[1])
    count = 0
    for b in bags:
        x = b.astype(np.float64)
        total += x.sum(0)
        total_sq += (x * x).sum(0)
        count += len(x)
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean * mean, 0)) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def pool(bag: np.ndarray, how: str) -> np.ndarray:
    """Raw (unstandardised) mean or max over tiles. Standardising afterwards is exact for both,
    because standardisation is a per-dimension increasing affine map."""
    x = bag.astype(np.float32)
    return x.mean(0) if how == "mean" else x.max(0)


def safe_auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


class PooledLR:
    """Logistic regression on a pooled slide embedding; C chosen on validation AUC."""

    def __init__(self, how: str):
        self.how = how

    def fit(self, pooled_fit, y_fit, pooled_val, y_val, mean, std, seed: int):
        self.mean, self.std = mean, std
        xf, xv = self._x(pooled_fit), self._x(pooled_val)
        best = None
        for c in config.LR_C_GRID:  # ascending, so ties keep the stronger regularisation
            lr = LogisticRegression(C=c, max_iter=5000, random_state=seed).fit(xf, y_fit)
            auc = safe_auc(y_val, lr.decision_function(xv))
            if best is None or auc > best[0]:
                best = (auc, c, lr)
        self.val_auc, self.choice, self.model = best
        return self

    def _x(self, pooled):
        return (np.asarray(pooled) - self.mean) / self.std

    def score(self, pooled) -> np.ndarray:
        return self.model.decision_function(self._x(pooled))


def _torch():
    import torch
    return torch


def build_abmil(dim: int):
    torch = _torch()
    nn = torch.nn
    p = config.ABMIL

    class GatedABMIL(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Sequential(nn.Linear(dim, p["hidden"]), nn.ReLU(), nn.Dropout(p["dropout"]))
            self.att_v = nn.Sequential(nn.Linear(p["hidden"], p["attention"]), nn.Tanh(), nn.Dropout(p["dropout"]))
            self.att_u = nn.Sequential(nn.Linear(p["hidden"], p["attention"]), nn.Sigmoid(), nn.Dropout(p["dropout"]))
            self.att_w = nn.Linear(p["attention"], 1)
            self.classify = nn.Linear(p["hidden"], 1)

        def forward(self, x):
            h = self.embed(x)
            a = torch.softmax(self.att_w(self.att_v(h) * self.att_u(h)).squeeze(-1), dim=0)
            return self.classify((a[:, None] * h).sum(0)).squeeze(-1), a

    return GatedABMIL()


class ABMILHead:
    """Gated attention MIL; the epoch with the best validation AUC (then lowest loss) is kept."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def fit(self, bags_fit, y_fit, bags_val, y_val, mean, std, seed: int, log=None):
        torch = _torch()
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.mean = torch.tensor(mean, device=self.device)
        self.std = torch.tensor(std, device=self.device)
        p = config.ABMIL
        model = build_abmil(len(mean)).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        loss_fn = torch.nn.BCEWithLogitsLoss()
        best = None
        for epoch in range(1, p["epochs"] + 1):
            model.train()
            for i in rng.permutation(len(bags_fit)):
                logit, _ = model(self._x(bags_fit[i]))
                loss = loss_fn(logit, torch.tensor(float(y_fit[i]), device=self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.model = model
            s = self.score(bags_val)
            auc = safe_auc(y_val, s)
            val_loss = float(np.mean(np.logaddexp(0, -s * (2 * np.asarray(y_val) - 1))))
            if best is None or (auc, -val_loss) > (best[0], -best[1]):
                best = (auc, val_loss, epoch, copy.deepcopy(model.state_dict()))
            if log:
                log(f"      epoch {epoch:2d}  val auc {auc:.3f}  val loss {val_loss:.3f}")
        self.val_auc, _, self.choice, state = best
        model.load_state_dict(state)
        self.model = model.eval()
        return self

    def _x(self, bag):
        torch = _torch()
        return (torch.as_tensor(np.asarray(bag, dtype=np.float32), device=self.device) - self.mean) / self.std

    def score(self, bags) -> np.ndarray:
        torch = _torch()
        self.model.eval()
        with torch.inference_mode():
            return np.array([float(self.model(self._x(b))[0]) for b in bags])

    def attention(self, bag) -> np.ndarray:
        """Attention weight of every tile of one bag (sums to 1)."""
        torch = _torch()
        self.model.eval()
        with torch.inference_mode():
            return self.model(self._x(bag))[1].float().cpu().numpy()

    def save(self, path) -> None:
        torch = _torch()
        torch.save(dict(state=self.model.state_dict(), mean=self.mean.cpu(), std=self.std.cpu(),
                        choice=self.choice, val_auc=self.val_auc), path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "ABMILHead":
        torch = _torch()
        saved = torch.load(path, map_location=device, weights_only=True)
        head = cls(device)
        head.mean, head.std = saved["mean"].to(device), saved["std"].to(device)
        head.choice, head.val_auc = saved["choice"], saved["val_auc"]
        head.model = build_abmil(len(head.mean)).to(device)
        head.model.load_state_dict(saved["state"])
        head.model.eval()
        return head
