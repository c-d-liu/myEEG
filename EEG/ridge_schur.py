# Shared core of the fast permutation test and the fast ablation fits for fixed-alpha
# ridge TRF models (CPU and GPU).
#
# Setting
# -------
# The design matrix is X = [A | B]: A holds the lagged columns of a fixed set of features
# (the "base"), B the lagged columns of additional features. Everything that depends only
# on A (and Y, the folds) is computed once and can be saved to disk (FixedPartStats); a
# model with any B is then evaluated from a few cross products of B with [A, Y, 1, B].
# The permutation test (permutation_fast.py) recomputes these cross products for every
# permutation of B; the ablation fits (ablation_fast.py) compute them once for all added
# features and pick column subsets.
#
# Math (per CV fold f, per alpha a; "tr" = training rows, "te" = test rows of the fold)
# ----
#   A_tr^T A_tr = U diag(lam) U^T                          (once per fold, FixedPartStats)
#   G = A_tr^T B_tr,  H = B_tr^T B_tr,  BY = B_tr^T Y_tr     (per added block / permutation)
#   D = 1 / (lam + a)
#   Schur complement of the ridge normal equations, solved in the eigenbasis of A_tr^T A_tr:
#     S    = H + aI - G~^T D G~                 with G~ = U^T G             (q x q)
#     W_b  = S^-1 (BY - G~^T D Y~)              with Y~ = U^T A_tr^T Y_tr
#     W~_a = D (Y~ - G~ W_b)                    (A-weights in the eigenbasis, W_a = U W~_a)
#   Test error from test-fold sufficient statistics (no test predictions needed):
#     SS_res = y^T y - 2 W~_a.(U^T A_te^T y) - 2 W_b.(B_te^T y)
#              + W~_a^T (U^T A_te^T A_te U) W~_a + 2 W~_a^T (U^T A_te^T B_te) W_b + W_b^T (B_te^T B_te) W_b
# With fit_intercept the training means are removed from every cross product (exactly as
# sklearn's Ridge(fit_intercept=True)); test errors are measured around the training means.
# With q = 0 (no B) this reduces to the plain ridge model on A.
#
# Raw per-fold sums (G_f = A_f^T B_f etc., f = rows of test fold f) are what gets stored:
# training statistics of fold f are "total - fold f", and centring only needs column sums.
#
# Everything runs through an array module `xp` (numpy or cupy) and the matching sparse
# module (scipy.sparse or cupyx.scipy.sparse), so the CPU and GPU versions share the code.
import json
import time
import numpy as np

FORMAT_VERSION = 1


# ---------------------------------------------------------------------- helpers
def get_backend(use_gpu):
    if use_gpu:
        import cupy
        import cupyx.scipy.sparse as sp
        return cupy, sp
    import scipy.sparse as sp
    return np, sp


def to_host(x):
    return x.get() if hasattr(x, 'get') else np.asarray(x)


def center(XtY, tx, ty, mx, my, n):
    """Cross product about given means: sum_i (x_i - mx)(y_i - my)^T, from the raw cross
    product XtY = sum_i x_i y_i^T and the raw sums tx, ty over the same n rows.
    Leading batch dimensions broadcast. With mx = my = 0 this returns XtY unchanged."""
    return (XtY - tx[..., :, None] * my[..., None, :] - mx[..., :, None] * ty[..., None, :]
            + n * mx[..., :, None] * my[..., None, :])


def fold_assignment(folds, n):
    """(n,) index of the test fold of every row. Test sets must partition the rows."""
    fold_of = np.full(n, -1)
    for f, (_, te) in enumerate(folds):
        fold_of[np.asarray(te)] = f
    if np.any(fold_of < 0):
        raise ValueError("test sets of the folds must cover every row")
    return fold_of


def channel_alphas(alphas, n_ch):
    """One alpha per channel from a scalar or a per-channel sequence."""
    alphas = np.asarray(alphas, dtype=np.float64).ravel()
    if alphas.size == 1:
        alphas = np.repeat(alphas, n_ch)
    if alphas.size != n_ch:
        raise ValueError(f"alphas must be a scalar or have one value per channel ({n_ch}), got {alphas.size}")
    return alphas


def _as_2d_float(Y):
    Y = np.asarray(Y, dtype=np.float64)
    return Y[:, None] if Y.ndim == 1 else Y


def _save_npz(path, kind, arrays, meta):
    np.savez(path, _kind=np.array(kind), _version=np.array(FORMAT_VERSION),
             _meta=np.array(json.dumps(meta if meta is not None else {})),
             **{k: to_host(v) for k, v in arrays.items()})


def _load_npz(path, kind):
    d = np.load(path, allow_pickle=False)
    if str(d['_kind']) != kind:
        raise ValueError(f"{path} holds {d['_kind']}, not {kind}")
    if int(d['_version']) != FORMAT_VERSION:
        raise ValueError(f"{path} has format version {int(d['_version'])}, expected {FORMAT_VERSION}")
    meta = json.loads(str(d['_meta']))
    arrays = {k: d[k] for k in d.files if not k.startswith('_')}
    return arrays, meta


def lagged_event_entries(series, rows, window):
    """
    Non-zero entries of the lagged matrix of event-like series, restricted to the kept rows.

    Row r of the lagged matrix covers series samples r .. r+window-1 and its column
    g*window + k holds series[g][r + k] (same layout as make_lagged_matrix).

    Args:
        series (list[np.ndarray]): full, unlagged 1-D series (same length).
        rows (np.ndarray): (n,) kept rows of the lagged matrix (np.flatnonzero(mask)).
        window (int): number of lags.
    Returns:
        ent_i (data row in 0..n-1), ent_col (column in 0..len(series)*window-1),
        ent_j (event index, events of all series concatenated), event_values (per series,
        values at its non-zero samples), event_slices (per series, its range of event indices).
    """
    n_lag_rows = len(series[0]) - window + 1
    pos = np.full(n_lag_rows, -1)
    pos[np.asarray(rows)] = np.arange(len(rows))
    ent_i, ent_col, ent_j, event_values, event_slices = [], [], [], [], []
    j0 = 0
    for g, b in enumerate(series):
        b = np.asarray(b, dtype=np.float64)
        if len(b) != len(series[0]):
            raise ValueError("all series must have the same length")
        s = np.flatnonzero(b)
        event_values.append(b[s])
        event_slices.append(slice(j0, j0 + len(s)))
        for k in range(window):
            r = s - k
            ok = (r >= 0) & (r < n_lag_rows)
            ok[ok] = pos[r[ok]] >= 0
            ent_i.append(pos[r[ok]])
            ent_col.append(np.full(ok.sum(), g * window + k))
            ent_j.append(j0 + np.flatnonzero(ok))
        j0 += len(s)
    return (np.concatenate(ent_i), np.concatenate(ent_col), np.concatenate(ent_j),
            event_values, event_slices)


def lagged_sparse_block(series, rows, window):
    """scipy CSR (n, window) lagged matrix of one event-like series at the kept rows,
    built without materialising the dense lagged matrix."""
    import scipy.sparse
    ent_i, ent_col, ent_j, vals, _ = lagged_event_entries([series], rows, window)
    return scipy.sparse.csr_matrix((vals[0][ent_j], (ent_i, ent_col)), shape=(len(rows), window))


# ---------------------------------------------------------------------- fixed part
class FixedPartStats:
    """
    Alpha-independent, per-fold statistics of the fixed design A: the eigendecomposition of
    every training covariance and the test-fold cross products, in that eigenbasis. This is
    the expensive part (one p_A x p_A eigh per fold) and is what gets saved and reused.

    Build with FixedPartStats.from_data(...) or FixedPartStats.load(path). The arrays live on
    the GPU if use_gpu (except fold_of, n_tr, n_te, sA, sY, which stay on the host).

    Disk size is about 2 * n_folds * p_A^2 * 8 bytes (U and the test covariance per fold).
    """
    _DEVICE = ('lam', 'U', 'Yt', 'yte', 'Ate', 'yy_te', 'uA_tr', 'uA_te', 'umA', 'mY', 'sY_tr', 'sY_te', 'sst')
    _HOST = ('fold_of', 'n_tr', 'n_te', 'sA', 'sY', 'fit_intercept')

    def __init__(self, arrays, meta=None, use_gpu=False):
        self.xp, self.sp = get_backend(use_gpu)
        self.use_gpu = use_gpu
        self.meta = dict(meta or {})
        for k in self._HOST:
            setattr(self, k, np.asarray(arrays[k]))
        for k in self._DEVICE:
            setattr(self, k, self.xp.asarray(arrays[k]))
        self.fit_intercept = bool(self.fit_intercept)
        self.n = len(self.fold_of)
        self.n_folds = len(self.n_tr)
        self.p_a = self.lam.shape[1]
        self.n_ch = self.yy_te.shape[1]

    # ------------------------------------------------------------ build
    @classmethod
    def from_data(cls, A, Y, folds, fit_intercept=True, use_gpu=False, meta=None, verbose=True):
        """
        Args:
            A (np.ndarray): (n, p_A) lagged columns of the fixed features, rows already masked.
            Y (np.ndarray): (n, n_channels) targets, same rows.
            folds (list[(train_idx, test_idx)]): CV splits over the n rows (test sets must
                partition the rows, as produced by stratified_group_kfold).
            fit_intercept (bool): unpenalised intercept like sklearn's Ridge (used by
                EEGData.fit_ridge_regression). False reproduces fit_fast_ridge_regression.
            use_gpu (bool): compute (and keep the result) on the GPU with CuPy.
            meta (dict): JSON-serialisable description stored with the statistics.
        """
        xp, _ = get_backend(use_gpu)
        t0 = time.perf_counter()
        Y = _as_2d_float(Y)
        n, p_a = A.shape
        n_ch = Y.shape[1]
        if len(Y) != n:
            raise ValueError("A and Y must have the same number of rows")
        fold_of = fold_assignment(folds, n)
        F = len(folds)
        sA_host = np.asarray(A, dtype=np.float64).sum(axis=0)
        sY_host = Y.sum(axis=0)

        # per-fold raw cross products of the fixed part (A is moved to the device fold by fold)
        C_f, AY_f, yy_f, sst_f, sA_f, sY_f = [], [], [], [], [], []
        for f in range(F):
            idx = np.flatnonzero(fold_of == f)
            Af = xp.asarray(A[idx], dtype=xp.float64)
            Yf = xp.asarray(Y[idx])
            C_f.append(Af.T @ Af)
            AY_f.append(Af.T @ Yf)
            sA_f.append(Af.sum(axis=0))
            sY_f.append(Yf.sum(axis=0))
            yy_f.append(xp.sum(Yf ** 2, axis=0))
            sst_f.append(xp.sum((Yf - Yf.mean(axis=0)) ** 2, axis=0))
            del Af, Yf
        C_tot, AY_tot, sA_tot, sY_tot = sum(C_f), sum(AY_f), sum(sA_f), sum(sY_f)
        if verbose:
            print(f"[fixed] covariances: {time.perf_counter() - t0:.1f}s "
                  f"(n={n}, p_A={p_a}, channels={n_ch}, folds={F})", flush=True)

        n_te = np.array([np.sum(fold_of == f) for f in range(F)])
        n_tr = n - n_te
        out = {k: xp.empty(s) for k, s in [
            ('lam', (F, p_a)), ('U', (F, p_a, p_a)), ('Yt', (F, p_a, n_ch)), ('yte', (F, p_a, n_ch)),
            ('Ate', (F, p_a, p_a)), ('yy_te', (F, n_ch)), ('uA_tr', (F, p_a)), ('uA_te', (F, p_a)),
            ('umA', (F, p_a)), ('mY', (F, n_ch)), ('sY_tr', (F, n_ch)), ('sY_te', (F, n_ch))]}
        for f in range(F):
            ntr, nte = int(n_tr[f]), int(n_te[f])
            sA_tr, sY_tr = sA_tot - sA_f[f], sY_tot - sY_f[f]
            mA = sA_tr / ntr if fit_intercept else xp.zeros(p_a)
            mY = sY_tr / ntr if fit_intercept else xp.zeros(n_ch)
            lam, U = xp.linalg.eigh(center(C_tot - C_f[f], sA_tr, sA_tr, mA, mA, ntr))
            out['lam'][f] = lam
            out['U'][f] = U
            out['Yt'][f] = U.T @ center(AY_tot - AY_f[f], sA_tr, sY_tr, mA, mY, ntr)
            out['yte'][f] = U.T @ center(AY_f[f], sA_f[f], sY_f[f], mA, mY, nte)
            out['Ate'][f] = U.T @ center(C_f[f], sA_f[f], sA_f[f], mA, mA, nte) @ U
            out['yy_te'][f] = yy_f[f] - 2 * mY * sY_f[f] + nte * mY ** 2
            out['uA_tr'][f] = U.T @ sA_tr
            out['uA_te'][f] = U.T @ sA_f[f]
            out['umA'][f] = U.T @ mA
            out['mY'][f] = mY
            out['sY_tr'][f] = sY_tr
            out['sY_te'][f] = sY_f[f]
            C_f[f] = None
            del lam, U
            if verbose:
                print(f"[fixed] eigendecomposition {f + 1}/{F}: {time.perf_counter() - t0:.1f}s", flush=True)
        del C_tot, AY_tot, C_f
        out['sst'] = xp.stack(sst_f)
        out.update(fold_of=fold_of, n_tr=n_tr, n_te=n_te, sA=sA_host, sY=sY_host,
                   fit_intercept=np.array(fit_intercept))
        return cls(out, meta=meta, use_gpu=use_gpu)

    # ------------------------------------------------------------ io
    def save(self, path):
        arrays = {k: getattr(self, k) for k in self._DEVICE + self._HOST}
        arrays['fit_intercept'] = np.array(self.fit_intercept)
        _save_npz(path, 'FixedPartStats', arrays, self.meta)

    @classmethod
    def load(cls, path, use_gpu=False):
        arrays, meta = _load_npz(path, 'FixedPartStats')
        return cls(arrays, meta=meta, use_gpu=use_gpu)

    # ------------------------------------------------------------ checks
    def check_data(self, A=None, Y=None, folds=None):
        """Raise if A, Y or the folds are not the ones these statistics were built from
        (compares shapes, column sums and fold assignments)."""
        if folds is not None and not np.array_equal(fold_assignment(folds, self.n), self.fold_of):
            raise ValueError("CV folds differ from those of the fixed-part statistics")
        for name, M, ref in (('A', A, self.sA), ('Y', Y, self.sY)):
            if M is None:
                continue
            M = _as_2d_float(M) if name == 'Y' else np.asarray(M, dtype=np.float64)
            if M.shape != (self.n, len(ref)):
                raise ValueError(f"{name} has shape {M.shape}, the fixed-part statistics expect {(self.n, len(ref))}")
            s = M.sum(axis=0)
            if not np.allclose(s, ref, rtol=1e-8, atol=1e-8 * max(1.0, np.abs(ref).max())):
                raise ValueError(f"{name} differs from the matrix the fixed-part statistics were built from")


# ---------------------------------------------------------------------- solver
def rotate_added(base, f, G_fold, G_tot):
    """A^T B of fold f in the eigenbasis of fold f's training covariance (uncentred):
    (U_f^T (G_tot - G_fold), U_f^T G_fold). Inputs (..., p_A, q) with any leading dims."""
    xp = base.xp
    U = base.U[f]
    p_a = base.p_a

    def rot(M):
        lead = M.shape[:-2]
        q = M.shape[-1]
        Mt = xp.moveaxis(M.reshape(-1, p_a, q), 1, 0).reshape(p_a, -1)
        return xp.moveaxis((U.T @ Mt).reshape(p_a, -1, q), 0, 1).reshape(*lead, p_a, q)

    return rot(G_tot - G_fold), rot(G_fold)


def schur_r2(base, alpha_sets, BY_f=None, sB_f=None, H_f=None, G_f=None, Gt=None):
    """
    Cross-validated R^2 of the ridge model on [A | B] for a batch of K added blocks B
    (all with q columns), for several alpha settings.

    Args:
        base (FixedPartStats): statistics of A.
        alpha_sets (array): (S, n_channels) alphas; each row is one setting.
        BY_f (F, K, q, ch), sB_f (F, K, q), H_f (F, K, q, q): raw per-fold sums
            B_f^T Y_f, column sums of B_f, B_f^T B_f (f = test rows of fold f).
        G_f (F, K, p_A, q): raw per-fold A_f^T B_f, or instead
        Gt (tuple): (Gt_tr, Gt_te), each (F, K, p_A, q), the output of rotate_added for
            every fold (pass this when the same columns are reused across calls).
        Omit all of them for the model on A alone (q = 0, K = 1).
    Returns:
        (S, F, K, n_channels) R^2 per alpha setting, fold and block (device array).
    """
    xp = base.xp
    F, p_a, ch = base.n_folds, base.p_a, base.n_ch
    alpha_sets = np.atleast_2d(np.asarray(alpha_sets, dtype=np.float64))
    if alpha_sets.shape[1] != ch:
        raise ValueError(f"alpha_sets must have {ch} columns (one per channel)")
    groups = [[(a, xp.asarray(np.flatnonzero(row == a))) for a in np.unique(row)] for row in alpha_sets]
    has_b = BY_f is not None and BY_f.shape[-2] > 0
    if has_b:
        K, q = BY_f.shape[1], BY_f.shape[2]
        BY_tot, H_tot, sB_tot = BY_f.sum(axis=0), H_f.sum(axis=0), sB_f.sum(axis=0)
        G_tot = G_f.sum(axis=0) if Gt is None else None
        eye = xp.eye(q)
    else:
        K = 1
    r2 = xp.zeros((len(alpha_sets), F, K, ch))
    for f in range(F):
        n_tr, n_te = int(base.n_tr[f]), int(base.n_te[f])
        if has_b:
            sB_tr, sB_te = sB_tot - sB_f[f], sB_f[f]                          # (K, q)
            mB = sB_tr / n_tr if base.fit_intercept else xp.zeros_like(sB_tr)
            umA = xp.broadcast_to(base.umA[f], (K, p_a))
            mY = xp.broadcast_to(base.mY[f], (K, ch))
            Gt_tr, Gt_te = rotate_added(base, f, G_f[f], G_tot) if Gt is None else (Gt[0][f], Gt[1][f])
            Gt_tr = center(Gt_tr, xp.broadcast_to(base.uA_tr[f], (K, p_a)), sB_tr, umA, mB, n_tr)
            Gt_te = center(Gt_te, xp.broadcast_to(base.uA_te[f], (K, p_a)), sB_te, umA, mB, n_te)
            H_tr = center(H_tot - H_f[f], sB_tr, sB_tr, mB, mB, n_tr)
            H_te = center(H_f[f], sB_te, sB_te, mB, mB, n_te)
            BY_tr = center(BY_tot - BY_f[f], sB_tr, xp.broadcast_to(base.sY_tr[f], (K, ch)), mB, mY, n_tr)
            BY_te = center(BY_f[f], sB_te, xp.broadcast_to(base.sY_te[f], (K, ch)), mB, mY, n_te)
            GtT = Gt_tr.transpose(0, 2, 1)                                    # (K, q, p_A)
        sst = base.sst[f]
        for s, alpha_groups in enumerate(groups):
            ss_res = xp.zeros((K, ch))
            for a, c in alpha_groups:
                D = 1.0 / (base.lam[f] + a)                                   # (p_A,)
                DY = D[:, None] * base.Yt[f][:, c]                            # (p_A, c)
                if has_b:
                    DG = D[None, :, None] * Gt_tr                             # (K, p_A, q)
                    S = H_tr + a * eye - GtT @ DG                             # (K, q, q)
                    rhs = BY_tr[:, :, c] - GtT @ DY                           # (K, q, c)
                    Wb = xp.linalg.solve(S, rhs)                              # (K, q, c)
                    Wa = DY[None] - DG @ Wb                                   # (K, p_A, c)
                else:
                    Wa = DY[None]                                             # (1, p_A, c)
                AWa = (base.Ate[f] @ Wa.transpose(1, 0, 2).reshape(p_a, -1)).reshape(p_a, K, -1).transpose(1, 0, 2)
                ss = (base.yy_te[f][c][None]
                      - 2 * xp.einsum('kpc,pc->kc', Wa, base.yte[f][:, c])
                      + xp.sum(Wa * AWa, axis=1))
                if has_b:
                    ss = ss + (- 2 * xp.sum(Wb * BY_te[:, :, c], axis=1)
                               + 2 * xp.sum(Wa * (Gt_te @ Wb), axis=1)
                               + xp.sum(Wb * (H_te @ Wb), axis=1))
                ss_res[:, c] = ss
            r2[s, f] = xp.where(sst[None] == 0, 0.0, 1 - ss_res / xp.where(sst == 0, 1.0, sst)[None])
    return r2
