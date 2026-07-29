# Shared helpers used by both the CPU and GPU ridge_cv backends, and by utils.py.
# Kept in their own module (rather than utils.py) so the backend modules can
# import them without creating a circular import with utils.py.
import numpy as np
from sklearn.model_selection import StratifiedKFold


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