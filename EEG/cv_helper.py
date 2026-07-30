# Shared helpers used by both the CPU and GPU ridge_cv backends, and by utils.py.
# Kept in their own module (rather than utils.py) so the backend modules can
# import them without creating a circular import with utils.py.
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .ridge_fast_cpu import fast_ridge_rsquared, optimal_alpha_rsquared, find_optimal_alphas_vectorized
from .utils import USE_GPU

def stratified_group_kfold(blocks, groups, n_splits=5, random_state=None):
    """
    Stratified k-fold split at the group level.
    Each group is assigned to exactly one fold, preserving class balance.
    """
    blocks = np.asarray(blocks)
    groups = np.asarray(groups)
    assert len(blocks) == len(groups), "Blocks and groups must have the same length"
    # unique groups + their label
    unique_groups, group_idx = np.unique(groups, return_index=True)
    group_labels = blocks[group_idx]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_group_idx, test_group_idx in skf.split(unique_groups, group_labels):
        train_groups = unique_groups[train_group_idx]
        test_groups = unique_groups[test_group_idx]

        train_mask = np.isin(groups, train_groups)
        test_mask = np.isin(groups, test_groups)

        yield np.where(train_mask)[0], np.where(test_mask)[0]


def to_cpu(obj):
    """Safely move CuPy arrays to CPU (NumPy) without crashing if they
    are already NumPy arrays, lists, etc."""
    if hasattr(obj, 'get'):
        return obj.get()
    return np.asarray(obj)

def ridge_cv_stratified_group(X, y, blocks, groups, alphas, n_splits=5, random_state=None,

                              USE_GPU=USE_GPU, optimize_alpha=False):

    '''Dispatcher for stratified-group ridge CV fitting.



    Use this function to pass a single alpha value to just fit a model with fixed alpha,

    or a list of alpha values to fit the model and find the best alpha per channel.

    It is not possible to pass per-channel alpha values.



    This function only validates inputs and dispatches to a backend implementation:

    ridge_cv_backend_gpu.fit_stratified_group_cv (cuML) if USE_GPU else

    ridge_cv_backend_cpu.fit_stratified_group_cv (scikit-learn).

    The backend is chosen per call from the USE_GPU argument (not just the module-level

    flag at import time), and results are always normalized back to host/NumPy arrays.

    '''

    X = np.asarray(X)

    y = np.asarray(y)

    assert len(X) == len(y) == len(blocks) == len(groups), f"X, y, blocks, and groups must have the same length.\nFound {(len(X), len(y), len(blocks), len(groups))}"



    if USE_GPU:

        from .ridge_gpu import fit_stratified_group_cv

    else:

        from .ridge_cpu import fit_stratified_group_cv



    avg_coefs_best, best_alpha_per_channel, best_metric_per_channel = fit_stratified_group_cv(

        X, y, blocks, groups, alphas,

        n_splits=n_splits, random_state=random_state, optimize_alpha=optimize_alpha,

    )



    # Normalize to host/NumPy regardless of which backend ran (no-op for CPU results).

    return to_cpu(avg_coefs_best), to_cpu(best_alpha_per_channel), to_cpu(best_metric_per_channel)

def ridge_cv_stratified_group_fast(X, y, blocks, groups, alphas, n_splits=5, random_state=None,
                                    USE_GPU=False, optimize_alpha=False, vectorized=True):
    """
    Stratified-group CV for Ridge regression using eigendecomposition-based R^2
    computation instead of fitting explicit models. Because no weight vectors are
    ever formed, this function returns ONLY R^2 (and, when optimize_alpha=True, the
    best alpha selected per channel).

    Which underlying helper from ridge_cv_fast is used depends on optimize_alpha and
    vectorized:
        - optimize_alpha=True,  vectorized=False -> fast_ridge_rsquared
              (looped per-alpha dict, one eigendecomposition per fold shared across
              alphas and channels)
        - optimize_alpha=True,  vectorized=True  -> find_optimal_alphas_vectorized
              (single 3D-tensor evaluation of every alpha x every channel per fold)
        - optimize_alpha=False                   -> optimal_alpha_rsquared
              (a single, possibly per-channel, alpha applied directly -- no search)

    NOTE: all three helpers use np.linalg.eigh, which expects NumPy arrays. If
    USE_GPU=True and X/y are GPU (e.g. cupy) arrays, swap np.linalg.eigh for the GPU
    library's equivalent (e.g. cupy.linalg.eigh) inside the relevant helper -- the
    rest of the math (matmuls, broadcasting) is API-compatible as-is.

    Args:
        X, y, blocks, groups: same as original function
        alphas: list of alphas to test. If optimize_alpha=True, this is the shared grid
            searched for every channel, and the best-per-channel alpha is selected
            afterwards. If optimize_alpha=False, either pass a single alpha (applied to
            all channels) or one alpha per channel (len(alphas) == y.shape[1]).
        n_splits, random_state: passed to stratified_group_kfold
        USE_GPU: if True, cast final outputs back to CPU via to_cpu()
        optimize_alpha: whether to search alphas and pick the best per channel
        vectorized: only relevant when optimize_alpha=True. If True, use the fully
            vectorized 3D-tensor search (find_optimal_alphas_vectorized) instead of
            the per-alpha dict loop (fast_ridge_rsquared). Ignored when
            optimize_alpha=False, since optimal_alpha_rsquared is always used there.

    Returns:
        dict with key "r2" (mean R^2 per channel across folds, shape [n_channels]),
        and additionally key "alpha" (best alpha per channel) when optimize_alpha=True.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, np.newaxis]
    assert len(X) == len(y) == len(blocks) == len(groups), \
        f"X, y, blocks, and groups must have the same length.\nFound {(len(X), len(y), len(blocks), len(groups))}"

    n_channels = y.shape[1]

    if optimize_alpha:
        alpha_list = list(alphas)
        print(f"Searching for best alpha per channel from {len(alpha_list)} candidates: {alpha_list}", flush=True)
        alpha_arr = np.array(alpha_list)

        if vectorized:
            # Evaluate every alpha x every channel in one shot per fold via
            # find_optimal_alphas_vectorized, then aggregate the R2 grids across
            # folds before picking the winning alpha per channel (so the selection
            # is based on mean CV performance, not a single fold).
            r2_grid_folds = []  # each entry: [n_channels, n_alphas]

            for train_idx, test_idx in stratified_group_kfold(blocks, groups,
                                                                n_splits=n_splits,
                                                                random_state=random_state):
                fold_result = find_optimal_alphas_vectorized(X[train_idx], y[train_idx],
                                                               X[test_idx], y[test_idx],
                                                               alpha_arr)
                r2_grid_folds.append(fold_result["r2_grid"])

            mean_r2_grid = np.mean(np.stack(r2_grid_folds, axis=0), axis=0)  # [n_channels, n_alphas]

            best_alpha_indices = np.argmax(mean_r2_grid, axis=1)
            best_alpha_per_channel = alpha_arr[best_alpha_indices]
            best_r2_per_channel = mean_r2_grid[np.arange(n_channels), best_alpha_indices]

        else:
            # r2_per_alpha[alpha] collects one [n_channels] array per fold
            r2_per_alpha = {alpha: [] for alpha in alpha_list}

            for train_idx, test_idx in stratified_group_kfold(blocks, groups,
                                                                n_splits=n_splits,
                                                                random_state=random_state):
                fold_r2 = fast_ridge_rsquared(X[train_idx], y[train_idx],
                                               X[test_idx], y[test_idx],
                                               alpha_list)
                for alpha in alpha_list:
                    r2_per_alpha[alpha].append(fold_r2[alpha])

            # mean R2 per channel, per alpha
            mean_r2_per_alpha = {
                alpha: np.mean(np.stack(r2_per_alpha[alpha], axis=0), axis=0)  # [n_channels]
                for alpha in alpha_list
            }

            # ------------------------
            # Select best alpha per channel
            # ------------------------
            all_mean_r2 = np.stack([mean_r2_per_alpha[a] for a in alpha_list], axis=0)  # [n_alphas, n_channels]
            best_alpha_indices = np.argmax(all_mean_r2, axis=0)
            best_alpha_per_channel = alpha_arr[best_alpha_indices]
            best_r2_per_channel = np.max(all_mean_r2, axis=0)

        if USE_GPU:
            best_alpha_per_channel = to_cpu(best_alpha_per_channel)
            best_r2_per_channel = to_cpu(best_r2_per_channel)

        return {"r2": best_r2_per_channel, "alpha": best_alpha_per_channel}

    else:
        if len(alphas) == 1:
            print(f"Using fixed alpha: {alphas[0]}", flush=True)
            alphas = list(alphas) * n_channels
        else:
            print(f"Using fixed per-channel alphas: {alphas}", flush=True)
            assert len(alphas) == n_channels, \
                "If optimize_alpha is False, the length of alphas must match the number of channels in y"

        # No search needed -- optimal_alpha_rsquared applies the (per-channel) alpha
        # directly via a vectorized regularization matrix, without looping over
        # channels or unique alphas.
        alphas_arr = np.array(alphas)
        fold_r2_list = []

        for train_idx, test_idx in stratified_group_kfold(blocks, groups,
                                                            n_splits=n_splits,
                                                            random_state=random_state):
            fold_r2 = optimal_alpha_rsquared(X[train_idx], y[train_idx],
                                              X[test_idx], y[test_idx],
                                              alphas_arr)
            fold_r2_list.append(fold_r2)

        r2_scores = np.stack(fold_r2_list, axis=0)  # [n_folds, n_channels]
        mean_r2_channels = np.mean(r2_scores, axis=0)  # [n_channels]

        if USE_GPU:
            mean_r2_channels = to_cpu(mean_r2_channels)

        return {"r2": mean_r2_channels, "alpha": alphas_arr}
