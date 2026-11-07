# GPU (CuPy) backend for ridge_cv_stratified_group_fast.
#
# Same math as ridge_fast_cpu (eigendecomposition of X_train^T X_train, no intercept,
# no centering), but organised so the full design matrix never has to live on the GPU:
#
#   1. One streaming pass over X (row chunks, host -> device) accumulates per-fold
#      sufficient statistics  C_f = X_f^T X_f  and  R_f = X_f^T Y_f.
#      The training statistics of fold f are then  C - C_f  and  R - R_f, so the
#      expensive X^T X product is done once over the data instead of once per fold.
#   2. Per fold: eigh(C_train) on the GPU, build ridge weights for every
#      (channel, alpha) pair at once, and stream the test rows to accumulate SS_res.
#
# All array operations go through an array module `xp`, so the same code runs with
# xp=numpy (useful for testing on machines without a GPU) or xp=cupy.
import numpy as np


def _get_xp(xp):
    if xp is not None:
        return xp
    import cupy
    return cupy


def _iter_chunks(idx, chunk_size):
    for start in range(0, len(idx), chunk_size):
        yield idx[start:start + chunk_size]


def ridge_r2_grid_streaming(X, y, folds, alpha_mat, chunk_size=8192, dtype=np.float64, xp=None, verbose=True):
    """
    Compute validation R^2 for every fold, channel and alpha.

    Args:
        X (np.ndarray): host design matrix, shape (n_samples, n_features)
        y (np.ndarray): host targets, shape (n_samples, n_channels)
        folds (list[(train_idx, test_idx)]): CV splits (test sets must partition
            a subset of rows; train = all other fold rows, as produced by
            stratified_group_kfold)
        alpha_mat (np.ndarray): shape (n_channels, n_alphas). Row c holds the alphas
            evaluated for channel c (a shared grid is just broadcast to every row).
        chunk_size (int): number of rows moved to the GPU at once. Device memory used
            for the chunk is about chunk_size * n_features * itemsize.
        dtype: computation dtype. float64 (default) is recommended: the train
            covariance is obtained by subtraction (C - C_f), which loses precision in
            float32. float32 is much faster on consumer GPUs if you accept that.
        xp: array module (cupy by default; numpy for CPU testing)

    Returns:
        np.ndarray: R^2 of shape (n_folds, n_channels, n_alphas), on host.
    """
    xp = _get_xp(xp)
    to_host = np.asarray if xp is np else xp.asnumpy

    X = np.asarray(X)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, np.newaxis]
    n_features = X.shape[1]
    n_channels = y.shape[1]
    alpha_mat = np.broadcast_to(np.asarray(alpha_mat, dtype=np.float64), (n_channels, np.shape(alpha_mat)[-1]))
    n_alphas = alpha_mat.shape[1]
    test_sets = [np.sort(np.asarray(test_idx)) for _, test_idx in folds]

    # ---- Pass 1: per-fold sufficient statistics (kept on host to save device memory)
    C_folds, R_folds = [], []
    C_total = xp.zeros((n_features, n_features), dtype=dtype)
    R_total = xp.zeros((n_features, n_channels), dtype=dtype)
    for f, test_idx in enumerate(test_sets):
        C_f = xp.zeros((n_features, n_features), dtype=dtype)
        R_f = xp.zeros((n_features, n_channels), dtype=dtype)
        for rows in _iter_chunks(test_idx, chunk_size):
            Xc = xp.asarray(X[rows], dtype=dtype)
            Yc = xp.asarray(y[rows], dtype=dtype)
            C_f += Xc.T @ Xc
            R_f += Xc.T @ Yc
            del Xc, Yc
        C_total += C_f
        R_total += R_f
        C_folds.append(to_host(C_f))
        R_folds.append(to_host(R_f))
        del C_f, R_f
        if verbose:
            print(f"[gpu-fast] covariance for fold {f + 1}/{len(test_sets)} done", flush=True)

    alpha_dev = xp.asarray(alpha_mat, dtype=dtype)  # (channels, alphas)
    r2 = np.empty((len(test_sets), n_channels, n_alphas))

    # ---- Pass 2: per fold, eigendecomposition + streamed validation error
    for f, test_idx in enumerate(test_sets):
        C_train = C_total - xp.asarray(C_folds[f])
        R_train = R_total - xp.asarray(R_folds[f])
        eigvals, U = xp.linalg.eigh(C_train)
        del C_train
        Usr = U.T @ R_train                                              # (features, channels)
        D = 1.0 / (eigvals[:, None, None] + alpha_dev[None, :, :])      # (features, channels, alphas)
        W = U @ (D * Usr[:, :, None]).reshape(n_features, n_channels * n_alphas)
        del U, D, Usr, R_train

        ss_res = xp.zeros((n_channels, n_alphas), dtype=dtype)
        for rows in _iter_chunks(test_idx, chunk_size):
            Xc = xp.asarray(X[rows], dtype=dtype)
            Yc = xp.asarray(y[rows], dtype=dtype)
            P = (Xc @ W).reshape(len(rows), n_channels, n_alphas)
            ss_res += xp.sum((Yc[:, :, None] - P) ** 2, axis=0)
            del Xc, Yc, P
        del W

        y_val = y[test_idx]
        ss_tot = np.sum((y_val - y_val.mean(axis=0)) ** 2, axis=0)     # (channels,), host float64
        ss_res = to_host(ss_res).astype(np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            r2[f] = np.where(ss_tot[:, None] == 0, 0.0, 1 - ss_res / ss_tot[:, None])
        if verbose:
            print(f"[gpu-fast] fold {f + 1}/{len(test_sets)} evaluated", flush=True)

    if xp is not np:
        xp.get_default_memory_pool().free_all_blocks()
    return r2
