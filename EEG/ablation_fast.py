# Fast ablation fits for ridge TRF models (CPU and GPU).
#
# Models of the form model(base, <subset of added features>) all share the base design A.
# The expensive part, one eigendecomposition of A's training covariance per fold, is a
# FixedPartStats (ridge_schur.py). The added features enter only through cross products
# of their lagged columns B with [A, Y, 1, B], computed ONCE for all added features
# (AddedFeatureStats); every submodel picks its columns from those and is solved with the
# Schur complement (ridge_schur.schur_r2): per model the cost is O(F * p_A * q^2) per alpha,
# with no p_A x p_A factorisation and no pass over the data.
#
# Typical use (see also EEGData.ablation_fast):
#   base  = FixedPartStats.from_data(A, Y, folds, fit_intercept=True)      ; base.save('base.npz')
#   added = AddedFeatureStats.from_data(base, A, Y, {'a': Ba, 'b': Bb, 'c': Bc}); added.save('added.npz')
#   abl   = FastRidgeAblation(base, added)
#   res   = abl.run([['a', 'b', 'c'], ['a', 'b'], ['b', 'c'], ['a', 'c'], ['c']], alphas=1000)
#
# Both intermediate files can be reloaded later (FixedPartStats.load / AddedFeatureStats.load)
# so new combinations and alphas are evaluated without touching the data.
import time
import numpy as np

from .ridge_schur import (FixedPartStats, get_backend, to_host, channel_alphas, rotate_added,
                          schur_r2, _save_npz, _load_npz, _as_2d_float)


def _issparse(M):
    import scipy.sparse
    return scipy.sparse.issparse(M)


def _to_device(M, xp, sp):
    if _issparse(M):
        return M.tocsr() if xp is np else sp.csr_matrix(M.tocsr())
    return xp.asarray(M, dtype=xp.float64)


def _cross(X, Z, sp):
    """X^T Z as a dense array for any mix of dense / sparse X, Z (on one device)."""
    if sp.issparse(X):
        R = X.T @ Z
        return R.toarray() if sp.issparse(R) else R
    if sp.issparse(Z):
        return (Z.T @ X).T
    return X.T @ Z


def _colsum(X, xp, sp):
    return X.T @ xp.ones(X.shape[0]) if sp.issparse(X) else X.sum(axis=0)


class AddedFeatureStats:
    """
    Raw per-fold cross products of the added features' lagged columns B with A, Y, 1 and B:
        G_f  (F, p_A, q)  A_f^T B_f        BY_f (F, q, ch)  B_f^T Y_f
        sB_f (F, q)       column sums      H_f  (F, q, q)   B_f^T B_f
    (f = test rows of fold f), for all added features at once (columns concatenated in the
    order of `names`, `sizes[k]` columns each). Kept on the host.

    Build with AddedFeatureStats.from_data(base, A, Y, blocks) or AddedFeatureStats.load(path).
    Disk size is about F * q * (p_A + q + ch) * 8 bytes.

    Note: cross terms between features are part of H, so features cannot be appended to
    an existing AddedFeatureStats later; compute it with every feature you will need.
    """

    def __init__(self, arrays, meta=None):
        self.names = [str(s) for s in arrays['names']]
        self.sizes = [int(s) for s in arrays['sizes']]
        for k in ('G_f', 'BY_f', 'sB_f', 'H_f', 'fold_of', 'sA', 'sY'):
            setattr(self, k, np.asarray(arrays[k]))
        self.meta = dict(meta or {})
        offsets = np.r_[0, np.cumsum(self.sizes)]
        self.slices = {k: slice(int(offsets[i]), int(offsets[i + 1])) for i, k in enumerate(self.names)}
        self.q = int(offsets[-1])

    @classmethod
    def from_data(cls, base, A, Y, blocks, meta=None, verbose=True):
        """
        Args:
            base (FixedPartStats): statistics of the base design A (gives the folds and the
                device; A and Y are checked against it).
            A (np.ndarray): (n, p_A) base design, the same matrix base was built from.
            Y (np.ndarray): (n, n_channels) targets.
            blocks (dict[str, array]): lagged columns of each added feature at the same rows,
                (n, q_k) dense ndarray or scipy.sparse matrix (use sparse for event features,
                e.g. ridge_schur.lagged_sparse_block). Dict order sets the column order.
            meta (dict): JSON-serialisable description stored with the statistics.
        """
        xp, sp = base.xp, base.sp
        t0 = time.perf_counter()
        Y = _as_2d_float(Y)
        base.check_data(A, Y)
        names = list(blocks)
        mats = []
        for k in names:
            M = blocks[k]
            if M.shape[0] != base.n:
                raise ValueError(f"block {k!r} has {M.shape[0]} rows, expected {base.n}")
            mats.append(M.tocsr().astype(np.float64) if _issparse(M) else M)
        sizes = [M.shape[1] for M in mats]
        offsets = np.r_[0, np.cumsum(sizes)]
        sl = [slice(int(offsets[i]), int(offsets[i + 1])) for i in range(len(names))]
        F, p_a, ch, q = base.n_folds, base.p_a, base.n_ch, int(offsets[-1])
        G = np.empty((F, p_a, q))
        BY = np.empty((F, q, ch))
        sB = np.empty((F, q))
        H = np.empty((F, q, q))
        for f in range(F):
            idx = np.flatnonzero(base.fold_of == f)
            Af = xp.asarray(A[idx], dtype=xp.float64)
            Yf = xp.asarray(Y[idx])
            Bf = [_to_device(M[idx], xp, sp) for M in mats]
            for i, Bi in enumerate(Bf):
                G[f][:, sl[i]] = to_host(_cross(Bi, Af, sp)).T
                BY[f][sl[i]] = to_host(_cross(Bi, Yf, sp))
                sB[f][sl[i]] = to_host(_colsum(Bi, xp, sp))
                for j in range(i, len(Bf)):
                    Hij = to_host(_cross(Bi, Bf[j], sp))
                    H[f][sl[i], sl[j]] = Hij
                    H[f][sl[j], sl[i]] = Hij.T
            del Af, Yf, Bf
            if verbose:
                print(f"[added] fold {f + 1}/{F}: {time.perf_counter() - t0:.1f}s "
                      f"(features={names}, q={q})", flush=True)
        arrays = dict(names=np.array(names), sizes=np.array(sizes), G_f=G, BY_f=BY, sB_f=sB, H_f=H,
                      fold_of=base.fold_of, sA=base.sA, sY=base.sY)
        return cls(arrays, meta=meta)

    def save(self, path):
        arrays = {k: getattr(self, k) for k in ('G_f', 'BY_f', 'sB_f', 'H_f', 'fold_of', 'sA', 'sY')}
        arrays.update(names=np.array(self.names), sizes=np.array(self.sizes))
        _save_npz(path, 'AddedFeatureStats', arrays, self.meta)

    @classmethod
    def load(cls, path):
        arrays, meta = _load_npz(path, 'AddedFeatureStats')
        return cls(arrays, meta=meta)

    def columns(self, features):
        """Column indices of `features` (in the given order) within the concatenated B."""
        unknown = [f for f in features if f not in self.slices]
        if unknown:
            raise KeyError(f"features {unknown} are not in the added statistics ({self.names})")
        if not features:
            return np.zeros(0, dtype=int)
        return np.concatenate([np.arange(self.slices[f].start, self.slices[f].stop) for f in features])


class FastRidgeAblation:
    """
    Cross-validated R^2 of model(base, *features) for any subset of the added features.

    Args:
        base (FixedPartStats | str): statistics of the base design (or a path).
        added (AddedFeatureStats | str): statistics of the added features (or a path).
        use_gpu (bool): only used when base is given as a path.
    """

    def __init__(self, base, added, use_gpu=False, verbose=True):
        if isinstance(base, str):
            base = FixedPartStats.load(base, use_gpu=use_gpu)
        if isinstance(added, str):
            added = AddedFeatureStats.load(added)
        if not np.array_equal(added.fold_of, base.fold_of):
            raise ValueError("added and base statistics were computed with different CV folds")
        if (added.G_f.shape[1] != base.p_a or added.BY_f.shape[2] != base.n_ch
                or not np.allclose(added.sA, base.sA) or not np.allclose(added.sY, base.sY)):
            raise ValueError("added statistics were computed for a different base design or targets")
        self.base, self.added, self.verbose = base, added, verbose
        xp = base.xp
        t0 = time.perf_counter()
        # rotate all added columns into every fold's eigenbasis once; models only slice them
        G_tot = xp.asarray(added.G_f.sum(axis=0))
        F, p_a, q = base.n_folds, base.p_a, added.q
        self.Gt_tr = xp.empty((F, p_a, q))
        self.Gt_te = xp.empty((F, p_a, q))
        for f in range(F):
            self.Gt_tr[f], self.Gt_te[f] = rotate_added(base, f, xp.asarray(added.G_f[f]), G_tot)
        del G_tot
        self.BY_f, self.sB_f, self.H_f = (xp.asarray(added.BY_f), xp.asarray(added.sB_f), xp.asarray(added.H_f))
        if verbose:
            print(f"[ablation] added columns rotated: {time.perf_counter() - t0:.1f}s", flush=True)

    @property
    def features(self):
        return list(self.added.names)

    def _r2_sets(self, features, alpha_sets):
        """(S, F, n_channels) R^2 of model(base, *features) for each row of alpha_sets."""
        features = list(features)
        if len(set(features)) != len(features):
            raise ValueError(f"duplicate features in {features}")
        if not features:
            return schur_r2(self.base, alpha_sets)[:, :, 0]
        xp = self.base.xp
        c = xp.asarray(self.added.columns(features))
        r2 = schur_r2(self.base, alpha_sets,
                      BY_f=self.BY_f[:, c][:, None], sB_f=self.sB_f[:, c][:, None],
                      H_f=self.H_f[:, c][:, :, c][:, None],
                      Gt=(self.Gt_tr[:, :, c][:, None], self.Gt_te[:, :, c][:, None]))
        return r2[:, :, 0]

    def r2(self, features, alphas, per_fold=False):
        """R^2 of model(base, *features) with fixed alphas (scalar or one per channel).
        Returns (n_channels,) mean over folds, or (n_folds, n_channels) if per_fold."""
        r2 = to_host(self._r2_sets(features, channel_alphas(alphas, self.base.n_ch)[None])[0])
        return r2 if per_fold else r2.mean(axis=0)

    def r2_grid(self, features, alpha_grid, per_fold=False):
        """R^2 of model(base, *features) for every alpha of a grid (shared by all channels).
        Returns (n_alphas, n_channels) mean over folds, or (n_alphas, n_folds, n_channels)."""
        grid = np.asarray(alpha_grid, dtype=np.float64).ravel()
        r2 = to_host(self._r2_sets(features, np.repeat(grid[:, None], self.base.n_ch, axis=1)))
        return r2 if per_fold else r2.mean(axis=1)

    def fit(self, features, alphas=None, alpha_grid=None):
        """
        Evaluate model(base, *features), either with fixed `alphas` (scalar or per channel)
        or choosing per channel the best alpha of `alpha_grid` on the mean CV R^2 (like
        optimize_alpha=True elsewhere in the package).

        Returns dict with features, r2 (n_channels,), r2_folds (n_folds, n_channels),
        alpha (n_channels,).
        """
        if (alphas is None) == (alpha_grid is None):
            raise ValueError("pass exactly one of alphas and alpha_grid")
        if alpha_grid is None:
            alpha = channel_alphas(alphas, self.base.n_ch)
            r2_folds = self.r2(features, alpha, per_fold=True)
        else:
            grid = np.asarray(alpha_grid, dtype=np.float64).ravel()
            folds = self.r2_grid(features, grid, per_fold=True)                 # (n_alphas, F, ch)
            best = np.argmax(folds.mean(axis=1), axis=0)
            alpha = grid[best]
            r2_folds = folds[best, :, np.arange(self.base.n_ch)].T              # (F, ch)
        return {"features": list(features), "r2": r2_folds.mean(axis=0), "r2_folds": r2_folds, "alpha": alpha}

    def run(self, models, alphas=None, alpha_grid=None):
        """
        Fit several models. `models` is a list of added-feature lists (the base is always
        included; [] is the base model alone) or a dict name -> list. `alphas` may also be
        a dict name -> alphas to use different fixed alphas per model.

        Returns dict name -> result of fit(); default names are 'base+a+b' (or 'base').
        """
        if not isinstance(models, dict):
            models = {'+'.join(['base'] + list(m)): list(m) for m in models}
        out = {}
        for name, feats in models.items():
            t0 = time.perf_counter()
            a = alphas[name] if isinstance(alphas, dict) else alphas
            out[name] = self.fit(feats, alphas=a, alpha_grid=alpha_grid)
            if self.verbose:
                print(f"[ablation] {name}: mean R^2 {out[name]['r2'].mean():.5f} "
                      f"({time.perf_counter() - t0:.2f}s)", flush=True)
        return out
