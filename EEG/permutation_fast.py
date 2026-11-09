# Fast permutation test for fixed-alpha ridge TRF models (CPU and GPU).
#
# The design matrix is X = [A | B]: A holds the lagged columns of every feature that is
# NOT permuted (fixed across permutations), B the lagged columns of the permuted
# feature(s). A permutation reassigns the values of a feature among its events (its
# non-zero samples), exactly like EEGData.permute_column: event positions stay put, so
# the sparsity pattern of B is identical for every permutation and only its values move.
#
# The fixed part (eigendecompositions of A per fold) is a FixedPartStats (ridge_schur.py),
# which can be saved once and reused (e.g. by parallel jobs with different seeds). Per
# permutation only the cross products of B are recomputed and the Schur-complement solver
# of ridge_schur.schur_r2 is applied; see ridge_schur.py for the math.
#
# Sufficient statistics of B are computed from its sparse structure:
#   * B^T [A, Y, 1] for every fold and every permutation of a batch is ONE sparse x dense
#     product with a stacked sparse matrix whose rows are (permutation, fold, lag).
#   * B^T B uses the precomputed list of (row, lag1, lag2) pairs where both entries are
#     non-zero (events closer together than the TRF window).
import time
import numpy as np

from .ridge_schur import (FixedPartStats, get_backend, to_host, channel_alphas,
                          lagged_event_entries, schur_r2, center)

_center = center  # backwards compatibility


class FastRidgePermutation:
    """
    Precomputes everything that does not depend on the permutation, then evaluates the
    cross-validated R^2 for batches of permutations.

    Args:
        A (np.ndarray): (n, p_A) lagged columns of the fixed (non-permuted) features,
            rows already masked (same rows as the model's X).
        Y (np.ndarray): (n, n_channels) targets, same rows.
        rows (np.ndarray): (n,) index of each kept row in the lagged-matrix row space, i.e.
            np.where(mask)[0]. Row r of the lagged matrix covers series samples r .. r+q-1.
        series (list[np.ndarray]): one 1-D series per permuted feature (the full,
            unlagged column, e.g. eeg.data[feature].to_numpy()).
        window (int): number of lags q (columns per feature in the lagged matrix).
        folds (list[(train_idx, test_idx)]): CV splits over the n rows (test sets must
            partition the rows, as produced by stratified_group_kfold).
        alphas (float | array): one alpha for all channels or one per channel.
        fit_intercept (bool): fit an unpenalised intercept like sklearn's Ridge (used by
            EEGData.fit_ridge_regression). False reproduces fit_fast_ridge_regression.
            Ignored when `base` is given (the base's setting is used).
        use_gpu (bool): run on the GPU with CuPy.
        verbose (bool): print timing of the precomputation.
        base (FixedPartStats | str | None): precomputed statistics of A (or a path to a file
            saved with FixedPartStats.save). They are checked against A, Y and folds.
    """

    def __init__(self, A, Y, rows, series, window, folds, alphas, fit_intercept=True, use_gpu=False,
                 verbose=True, base=None):
        xp, sp = get_backend(use_gpu)
        self.xp, self.sp, self.use_gpu, self.verbose = xp, sp, use_gpu, verbose
        t0 = time.perf_counter()
        A = np.ascontiguousarray(A, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        if len(rows) != len(A):
            raise ValueError("A, Y and rows must have the same number of rows")

        if base is None:
            base = FixedPartStats.from_data(A, Y, folds, fit_intercept=fit_intercept, use_gpu=use_gpu,
                                            verbose=verbose)
        else:
            if isinstance(base, str):
                base = FixedPartStats.load(base, use_gpu=use_gpu)
            if base.use_gpu != use_gpu:
                raise ValueError("base was loaded for a different device than use_gpu")
            base.check_data(A, Y, folds)
        self.base = base
        self.fit_intercept = base.fit_intercept
        self.n, self.p_a, self.n_ch, self.n_folds, self.window = base.n, base.p_a, base.n_ch, base.n_folds, window
        self.alphas = channel_alphas(alphas, self.n_ch)

        self._build_structure(rows, series, window, base.fold_of)
        self.A_d, self.Y_d = xp.asarray(A), xp.asarray(Y)
        if verbose:
            print(f"[perm] sparse structure: {self.nnz} non-zeros in B ({self.nnz / (self.n * self.q):.2%} dense), "
                  f"{self.n_pairs} B^T B pairs; precomputation done in {time.perf_counter() - t0:.1f}s", flush=True)

    # ------------------------------------------------------------------ structure
    def _build_structure(self, rows, series, window, fold_of):
        ent_i, ent_col, ent_j, self.event_values, self.event_slices = lagged_event_entries(series, rows, window)
        self.m = sum(len(v) for v in self.event_values)
        self.q = len(series) * window
        q, n_folds = self.q, self.n_folds
        self.nnz = len(ent_i)
        if self.nnz > 0.25 * self.n * q:
            raise ValueError("The permuted feature is not sparse (more than 25% non-zeros after lagging); "
                             "the sparse permutation engine is designed for event features.")

        # Stacked sparse matrix (one permutation): row = fold * q + col, column = data row i
        srow = fold_of[ent_i] * q + ent_col
        order = np.lexsort((ent_i, srow))
        srow, self.sp_indices, self.sp_event = srow[order], ent_i[order].astype(np.int32), ent_j[order]
        self.sp_indptr = np.searchsorted(srow, np.arange(n_folds * q + 1)).astype(np.int64)
        self._csr_cache = {}

        # B^T B pairs: all ordered pairs of non-zeros sharing a data row
        order = np.argsort(ent_i, kind='stable')
        ei, ec, ej = ent_i[order], ent_col[order], ent_j[order]
        starts = np.flatnonzero(np.r_[True, ei[1:] != ei[:-1]])
        counts = np.diff(np.r_[starts, len(ei)])
        cnt_e = np.repeat(counts, counts)                # group size, per entry
        start_e = np.repeat(starts, counts)              # group start, per entry
        n_pairs = int(np.sum(counts.astype(np.int64) ** 2))
        if n_pairs > 3e8:
            raise ValueError(f"Too many B^T B pairs ({n_pairs}); the permuted feature is too dense.")
        first = np.repeat(np.arange(len(ei)), cnt_e)
        offset = np.arange(n_pairs) - np.repeat(np.cumsum(cnt_e) - cnt_e, cnt_e)
        second = np.repeat(start_e, cnt_e) + offset
        target = fold_of[ei[first]] * q * q + ec[first] * q + ec[second]
        self.pair_j1 = ej[first]
        self.pair_j2 = ej[second]
        self.n_pairs = n_pairs
        xp, sp = self.xp, self.sp
        # P^T: (folds*q*q, n_pairs) with a single 1 per column -> scatter-add as a sparse product
        self.PT = sp.csr_matrix(
            (xp.ones(n_pairs), (xp.asarray(target), xp.arange(n_pairs))),
            shape=(n_folds * q * q, n_pairs))
        self.pair_j1_d = xp.asarray(self.pair_j1)
        self.pair_j2_d = xp.asarray(self.pair_j2)
        self.sp_event_d = xp.asarray(self.sp_event)

    def _stacked_csr(self, V):
        """Sparse (K*folds*q, n) matrix of B^T for every permutation (V: (K, m) on device)."""
        xp, sp = self.xp, self.sp
        K = V.shape[0]
        if K not in self._csr_cache:
            nnz = self.nnz
            indptr = np.concatenate([self.sp_indptr[:-1] + p * nnz for p in range(K)] + [[K * nnz]])
            self._csr_cache[K] = (xp.asarray(np.tile(self.sp_indices, K)), xp.asarray(indptr))
        indices, indptr = self._csr_cache[K]
        data = V[:, self.sp_event_d].ravel()
        return sp.csr_matrix((data, indices, indptr), shape=(K * self.n_folds * self.q, self.n))

    # ------------------------------------------------------------------ evaluation
    def added_stats(self, V):
        """Raw per-fold sums of B for a batch of value assignments V (K, m):
        G_f (F, K, p_A, q), BY_f (F, K, q, ch), sB_f (F, K, q), H_f (F, K, q, q)."""
        xp = self.xp
        V = xp.asarray(np.atleast_2d(V) if isinstance(V, np.ndarray) else V, dtype=xp.float64)
        K = V.shape[0]
        F, q, p_a, ch = self.n_folds, self.q, self.p_a, self.n_ch
        Bt = self._stacked_csr(V)
        G_f = (Bt @ self.A_d).reshape(K, F, q, p_a).transpose(1, 0, 3, 2)   # (F, K, p_A, q)
        BY_f = (Bt @ self.Y_d).reshape(K, F, q, ch).transpose(1, 0, 2, 3)   # (F, K, q, ch)
        sB_f = (Bt @ xp.ones(self.n)).reshape(K, F, q).transpose(1, 0, 2)   # (F, K, q)
        del Bt
        prod = V[:, self.pair_j1_d] * V[:, self.pair_j2_d]                  # (K, n_pairs)
        H_f = (self.PT @ prod.T).T.reshape(K, F, q, q).transpose(1, 0, 2, 3)  # (F, K, q, q)
        return G_f, BY_f, sB_f, H_f

    def r2(self, V):
        """
        Cross-validated R^2 for a batch of value assignments.

        Args:
            V (array): (K, m) event values for K permutations (columns ordered as the
                events of each permuted feature, concatenated; see event_values).
        Returns:
            np.ndarray: (K, n_channels) mean-over-folds R^2.
        """
        G_f, BY_f, sB_f, H_f = self.added_stats(V)
        r2 = schur_r2(self.base, self.alphas[None], BY_f=BY_f, sB_f=sB_f, H_f=H_f, G_f=G_f)[0]
        return to_host(r2.mean(axis=0))

    # ------------------------------------------------------------------ permutations
    def permuted_values(self, rng, K):
        """(K, m) values: each permuted feature's event values shuffled independently
        (same scheme as EEGData.permute_column)."""
        V = np.empty((K, self.m))
        for vals, sl in zip(self.event_values, self.event_slices):
            V[:, sl] = rng.permuted(np.broadcast_to(vals, (K, len(vals))), axis=1)
        return V

    def original_values(self):
        return np.concatenate(self.event_values)[None, :]

    def run(self, n_permutations, seed=0, batch_size=None):
        """Observed R^2, null distribution and permutation p-values.

        Returns dict with
            r2:       (n_channels,) observed cross-validated R^2
            r2_null:  (n_permutations, n_channels)
            p_values: (n_channels,) (1 + #{null >= observed}) / (1 + n_permutations)
        """
        if batch_size is None:
            # keep the dense B^T A block (K * folds * q * p_A doubles) around 2 GB (GPU) / 0.5 GB (CPU)
            budget = 2e9 if self.use_gpu else 5e8
            batch_size = int(max(1, min(256, budget // (self.n_folds * self.q * self.p_a * 8))))
        rng = np.random.default_rng(seed)
        r2_obs = self.r2(self.original_values())[0]
        null = np.empty((n_permutations, self.n_ch))
        t0 = time.perf_counter()
        for start in range(0, n_permutations, batch_size):
            K = min(batch_size, n_permutations - start)
            null[start:start + K] = self.r2(self.permuted_values(rng, K))
            if self.verbose:
                done = start + K
                el = time.perf_counter() - t0
                print(f"[perm] {done}/{n_permutations} permutations, {el / done * 1e3:.1f} ms/permutation", flush=True)
        p = (1 + np.sum(null >= r2_obs[None], axis=0)) / (1 + n_permutations)
        return {"r2": r2_obs, "r2_null": null, "p_values": p, "alphas": self.alphas}
