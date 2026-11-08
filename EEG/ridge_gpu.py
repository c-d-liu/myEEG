# GPU backend for ridge_cv_stratified_group.
# Only imported when USE_GPU is True, so cuML is only pulled in when actually needed.
import gc

import numpy as np
from cuml.linear_model import Ridge
from cuml.metrics import r2_score, mean_squared_error

from .cv_helper import stratified_group_kfold, to_cpu


def _fit_eval_fold(X, y, train_idx, test_idx, alpha):
    """Fit one cuML Ridge on a fold and return host-side results only.

    Nothing from the fold is kept on the GPU afterwards: coefficients and scores are
    copied to NumPy, the model and predictions are deleted, and gc runs so device
    buffers are released before the next fold allocates its training copy.
    """
    model = Ridge(alpha=alpha, solver='lsmr')
    model.fit(X[train_idx], y[train_idx])
    coef = to_cpu(model.coef_)

    y_test = y[test_idx]
    y_pred = model.predict(X[test_idx])
    r2 = to_cpu(r2_score(y_test, y_pred, multioutput="raw_values"))
    mse = float(to_cpu(mean_squared_error(y_test, y_pred)))

    del model, y_pred
    gc.collect()
    return coef, r2, mse


def fit_stratified_group_cv(X, y, blocks, groups, alphas, n_splits=5, random_state=None,
                             optimize_alpha=False, dtype=np.float32):
    """GPU (cuML) implementation of the stratified-group ridge CV fit.
    Mirrors ridge_cv_backend_cpu.fit_stratified_group_cv in inputs/outputs; all
    returned arrays are NumPy.
    See ridge_cv_stratified_group in utils.py for the public-facing docstring.

    dtype: precision used for X and y on the GPU. float32 (default) halves device
        memory compared to float64; pass np.float64 if you need full precision.
    """
    # Cast once on the host so every per-fold GPU copy is in the reduced precision.
    X = np.asarray(X, dtype=dtype)
    y = np.asarray(y, dtype=dtype)

    if optimize_alpha:
        mean_r2_channels_per_alpha = {}
        avg_coefs_per_alpha = {}

        for alpha in alphas:
            print(f"Testing alpha: {alpha}", flush=True)
            r2_scores, fold_coefs = [], []

            for train_idx, test_idx in stratified_group_kfold(blocks, groups,
                                                                n_splits=n_splits,
                                                                random_state=random_state):
                coef, r2, _ = _fit_eval_fold(X, y, train_idx, test_idx, alpha)
                fold_coefs.append(coef)
                r2_scores.append(r2)

            r2_scores = np.stack(r2_scores, axis=0)                          # [n_folds, n_channels]
            mean_r2_channels_per_alpha[alpha] = np.mean(r2_scores, axis=0)   # [n_channels]
            avg_coefs_per_alpha[alpha] = np.mean(np.stack(fold_coefs, axis=0), axis=0)  # [n_targets, n_features]

        # ------------------------
        # Select best alpha per channel
        # ------------------------
        all_mean_r2 = np.stack([mean_r2_channels_per_alpha[a] for a in alphas], axis=0)  # [n_alphas, n_targets]
        best_alpha_indices = np.argmax(all_mean_r2, axis=0)
        alphas_arr = np.asarray(alphas)
        best_alpha_per_channel = alphas_arr[best_alpha_indices]
        best_r2_per_channel = np.max(all_mean_r2, axis=0)

        # Select the coefficient row for each target using its best alpha
        n_targets, n_features = y.shape[1], X.shape[1]
        avg_coefs_best = np.zeros((n_targets, n_features))
        for t in range(n_targets):
            avg_coefs_best[t] = avg_coefs_per_alpha[alphas[best_alpha_indices[t]]][t]

        return avg_coefs_best, best_alpha_per_channel, best_r2_per_channel

    else:
        if len(alphas) == 1:
            print(f"Using fixed alpha: {alphas[0]}", flush=True)
            alphas = list(alphas) * y.shape[1]  # replicate the single alpha for all channels
        else:
            assert len(alphas) == y.shape[1], "If optimize_alpha is False, the length of alphas must match the number of channels in y"

        r2_scores, fold_coefs = [], []
        for train_idx, test_idx in stratified_group_kfold(blocks, groups,
                                                            n_splits=n_splits,
                                                            random_state=random_state):
            coef, r2, _ = _fit_eval_fold(X, y, train_idx, test_idx, alphas)
            fold_coefs.append(coef)
            r2_scores.append(r2)

        r2_scores = np.stack(r2_scores, axis=0)        # [n_folds, n_channels]
        mean_r2_channels = np.mean(r2_scores, axis=0)  # [n_channels]
        avg_coefs_best = np.mean(np.stack(fold_coefs, axis=0), axis=0)  # [n_channels, n_features]

        return avg_coefs_best, np.asarray(alphas), mean_r2_channels
