# Fast permutation test for fixed-alpha ridge TRF models (CPU and GPU).
#
# Setting
# -------
# The design matrix is X = [A | B]: A holds the lagged columns of every feature that is
# NOT permuted (fixed across permutations), B the lagged columns of the permuted
# feature(s). A permutation reassigns the values of a feature among its events (its
# non-zero samples), exactly like EEGData.permute_column: event positions stay put, so
# the sparsity pattern of B is identical for every permutation and only its values move.
#
# Math (per CV fold f, per alpha a; "tr" = training rows, "te" = test rows of the fold)
# ----
#   A_tr^T A_tr = U diag(lam) U^T                          (once per fold)
#   G = A_tr^T B_tr,  H = B_tr^T B_tr,  BY = B_tr^T Y_tr     (per permutation)
#   D = 1 / (lam + a)
#   Schur complement of the ridge normal equations, solved in the eigenbasis of A_tr^T A_tr:
#     S    = H + aI - G~^T D G~                 with G~ = U^T G             (q x q)
#     W_b  = S^-1 (BY - G~^T D Y~)              with Y~ = U^T A_tr^T Y_tr
#     W~_a = D (Y~ - G~ W_b)                    (A-weights in the eigenbasis, W_a = U W~_a)
#   Test error from test-fold sufficient statistics (no test predictions needed):
#     SS_res = y^T y - 2 W~_a.(U^T A_te^T y) - 2 W_b.(B_te^T y)
#              + W~_a^T (U^T A_te^T A_te U) W~_a + 2 W~_a^T (U^T A_te^T B_te) W_b + W_b^T (B_te^T B_te) W_b
# Nothing of size p x p is factorised per permutation; the per-permutation cost is a
# sparse product B^T A (nnz(B) * p_A) plus a few dense products of size p_A^2 * q.
#
# Sufficient statistics of B are computed from its sparse structure:
#   * B^T [A, Y] for every fold and every permutation of a batch is ONE sparse x dense
#     product with a stacked sparse matrix whose rows are (permutation, fold, lag).
#   * B^T B uses the precomputed list of (row, lag1, lag2) pairs where both entries are
#     non-zero (events closer together than the TRF window).
#
# Everything runs through an array module `xp` (numpy or cupy) and the matching sparse
# module (scipy.sparse or cupyx.scipy.sparse), so the CPU and GPU versions share the code.
import time
import numpy as np


def _center(XtY, tx, ty, mx, my, n):
    """Cross product about given means: sum_i (x_i - mx)(y_i - my)^T, from the raw cross
    product XtY = sum_i x_i y_i^T and the raw sums tx, ty over the same n rows.
    Leading batch dimensions broadcast. With mx = my = 0 this returns XtY unchanged."""
    return (XtY - tx[..., :, None] * my[..., None, :] - mx[..., :, None] * ty[..., None, :]
            + n * mx[..., :, None] * my[..., None, :])


def _backend(use_gpu):
    if use_gpu:
        import cupy
        import cupyx.scipy.sparse as sp
        return cupy, sp
    import scipy.sparse as sp
    return np, sp


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
        use_gpu (bool): run on the GPU with CuPy.
        verbose (bool): print timing of the precomputation.
    """

    def __init__(self, A, Y, rows, series, window, folds, alphas, fit_intercept=True, use_gpu=False, verbose=True):
        xp, sp = _backend(use_gpu)
        self.xp, self.sp, self.use_gpu, self.verbose = xp, sp, use_gpu, verbose
        t0 = time.perf_counter()

        A = np.ascontiguousarray(A, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        n, p_a = A.shape
        n_ch = Y.shape[1]
        rows = np.asarray(rows)
        assert len(rows) == n == len(Y), "A, Y and rows must have the same number of rows"
        alphas = np.asarray(alphas, dtype=np.float64).ravel()
        if alphas.size == 1:
            alphas = np.repeat(alphas, n_ch)
        assert alphas.size == n_ch, "alphas must be a scalar or have one value per channel"

        # ---- folds: every row belongs to exactly one test set
        n_folds = len(folds)
        fold_of = np.full(n, -1)
        for f, (_, te) in enumerate(folds):
            fold_of[np.asarray(te)] = f
        assert np.all(fold_of >= 0), "test sets of the folds must cover every row"
        self.n, self.p_a, self.n_ch, self.n_folds, self.window = n, p_a, n_ch, n_folds, window
        self.alphas = alphas
        self.alpha_groups = [(a, np.where(alphas == a)[0]) for a in np.unique(alphas)]

        # ---- per-fold statistics of the fixed part
        A_d = xp.asarray(A)
        Y_d = xp.asarray(Y)
        fold_rows = [xp.asarray(np.where(fold_of == f)[0]) for f in range(n_folds)]
        C_f, AY_f, yy_f, sst_f, sA_f, sY_f = [], [], [], [], [], []
        for f in range(n_folds):
            Af = A_d[fold_rows[f]]
            Yf = Y_d[fold_rows[f]]
            C_f.append(Af.T @ Af)
            AY_f.append(Af.T @ Yf)
            sA_f.append(Af.sum(axis=0))
            sY_f.append(Yf.sum(axis=0))
            yy_f.append(xp.sum(Yf ** 2, axis=0))
            sst_f.append(xp.sum((Yf - Yf.mean(axis=0)) ** 2, axis=0))
            del Af, Yf
        C_tot, AY_tot, sA_tot, sY_tot = sum(C_f), sum(AY_f), sum(sA_f), sum(sY_f)
        if verbose:
            print(f"[perm] fixed-part covariances: {time.perf_counter() - t0:.1f}s "
                  f"(n={n}, p_A={p_a}, channels={n_ch}, folds={n_folds})", flush=True)

        # Per fold: everything in the eigenbasis of the (centred) training covariance of A.
        # With fit_intercept the training means (mA, mY, and per permutation mB) are removed,
        # exactly as sklearn's Ridge(fit_intercept=True) does; test errors are measured
        # around the training means too (prediction = training mean + centred model).
        self.fit_intercept = fit_intercept
        self.n_te = [len(r) for r in fold_rows]
        self.n_tr = [n - m for m in self.n_te]
        self.lam, self.U, self.Yt, self.yte, self.Ate, self.yy_te = [], [], [], [], [], []
        self.uA_tr, self.uA_te, self.umA, self.mY, self.sY_tr, self.sY_te = [], [], [], [], [], []
        for f in range(n_folds):
            n_tr, n_te = self.n_tr[f], self.n_te[f]
            sA_tr, sY_tr = sA_tot - sA_f[f], sY_tot - sY_f[f]
            mA = sA_tr / n_tr if fit_intercept else xp.zeros(p_a)
            mY = sY_tr / n_tr if fit_intercept else xp.zeros(n_ch)
            lam, U = xp.linalg.eigh(_center(C_tot - C_f[f], sA_tr, sA_tr, mA, mA, n_tr))
            self.lam.append(lam)
            self.U.append(U)
            self.Yt.append(U.T @ _center(AY_tot - AY_f[f], sA_tr, sY_tr, mA, mY, n_tr))   # (p_A, ch)
            self.yte.append(U.T @ _center(AY_f[f], sA_f[f], sY_f[f], mA, mY, n_te))       # (p_A, ch)
            self.Ate.append(U.T @ _center(C_f[f], sA_f[f], sA_f[f], mA, mA, n_te) @ U)    # (p_A, p_A)
            self.yy_te.append(yy_f[f] - 2 * mY * sY_f[f] + n_te * mY ** 2)                # (ch,)
            self.uA_tr.append(U.T @ sA_tr)
            self.uA_te.append(U.T @ sA_f[f])
            self.umA.append(U.T @ mA)
            self.mY.append(mY)
            self.sY_tr.append(sY_tr)
            self.sY_te.append(sY_f[f])
            C_f[f] = None
        del C_tot, AY_tot, C_f
        self.sst_f = sst_f
        if verbose:
            print(f"[perm] eigendecompositions: {time.perf_counter() - t0:.1f}s", flush=True)

        # ---- sparse structure of B
        self._build_structure(rows, series, window, fold_of)
        self.A_d, self.Y_d = A_d, Y_d
        if verbose:
            print(f"[perm] sparse structure: {self.nnz} non-zeros in B ({self.nnz / (n * self.q):.2%} dense), "
                  f"{self.n_pairs} B^T B pairs; precomputation done in {time.perf_counter() - t0:.1f}s", flush=True)

    # ------------------------------------------------------------------ structure
    def _build_structure(self, rows, series, window, fold_of):
        n_lag_rows = len(series[0]) - window + 1
        pos = np.full(n_lag_rows, -1)
        pos[rows] = np.arange(len(rows))

        ent_i, ent_col, ent_j = [], [], []
        self.event_values, self.event_slices = [], []
        j0 = 0
        for g, b in enumerate(series):
            b = np.asarray(b, dtype=np.float64)
            assert len(b) == len(series[0]), "all permuted series must have the same length"
            s = np.flatnonzero(b)
            self.event_values.append(b[s])
            self.event_slices.append(slice(j0, j0 + len(s)))
            for k in range(window):
                r = s - k
                ok = (r >= 0) & (r < n_lag_rows)
                ok[ok] = pos[r[ok]] >= 0
                ent_i.append(pos[r[ok]])
                ent_col.append(np.full(ok.sum(), g * window + k))
                ent_j.append(j0 + np.flatnonzero(ok))
            j0 += len(s)
        self.m = j0
        self.q = len(series) * window
        q, n_folds = self.q, self.n_folds
        ent_i = np.concatenate(ent_i)
        ent_col = np.concatenate(ent_col)
        ent_j = np.concatenate(ent_j)
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
    def r2(self, V):
        """
        Cross-validated R^2 for a batch of value assignments.

        Args:
            V (array): (K, m) event values for K permutations (columns ordered as the
                events of each permuted feature, concatenated; see event_values).
        Returns:
            np.ndarray: (K, n_channels) mean-over-folds R^2.
        """
        xp = self.xp
        V = xp.asarray(np.atleast_2d(V) if isinstance(V, np.ndarray) else V, dtype=xp.float64)
        K = V.shape[0]
        F, q, p_a, ch = self.n_folds, self.q, self.p_a, self.n_ch

        # B^T A, B^T Y and column sums of B for every (permutation, fold, lag)
        Bt = self._stacked_csr(V)
        G_f = (Bt @ self.A_d).reshape(K, F, q, p_a).transpose(1, 0, 3, 2)   # (F, K, p_A, q)
        BY_f = (Bt @ self.Y_d).reshape(K, F, q, ch).transpose(1, 0, 2, 3)   # (F, K, q, ch)
        sB_f = (Bt @ xp.ones(self.n)).reshape(K, F, q).transpose(1, 0, 2)   # (F, K, q)
        del Bt
        # B^T B via pair list
        prod = V[:, self.pair_j1_d] * V[:, self.pair_j2_d]                  # (K, n_pairs)
        H_f = (self.PT @ prod.T).T.reshape(K, F, q, q).transpose(1, 0, 2, 3)  # (F, K, q, q)
        del prod

        G_tot, BY_tot, H_tot, sB_tot = G_f.sum(axis=0), BY_f.sum(axis=0), H_f.sum(axis=0), sB_f.sum(axis=0)
        r2 = xp.zeros((K, ch))
        eye = xp.eye(q)
        for f in range(F):
            U = self.U[f]
            n_tr, n_te = self.n_tr[f], self.n_te[f]
            sB_tr, sB_te = sB_tot - sB_f[f], sB_f[f]                          # (K, q)
            mB = sB_tr / n_tr if self.fit_intercept else xp.zeros_like(sB_tr)
            umA = xp.broadcast_to(self.umA[f], (K, p_a))
            mY = xp.broadcast_to(self.mY[f], (K, ch))
            # rotate A^T B (train and test part) into the eigenbasis of fold f, then centre
            Gt_tr = (U.T @ (G_tot - G_f[f]).transpose(1, 0, 2).reshape(p_a, K * q)).reshape(p_a, K, q).transpose(1, 0, 2)
            Gt_te = (U.T @ G_f[f].transpose(1, 0, 2).reshape(p_a, K * q)).reshape(p_a, K, q).transpose(1, 0, 2)
            Gt_tr = _center(Gt_tr, xp.broadcast_to(self.uA_tr[f], (K, p_a)), sB_tr, umA, mB, n_tr)
            Gt_te = _center(Gt_te, xp.broadcast_to(self.uA_te[f], (K, p_a)), sB_te, umA, mB, n_te)
            H_tr = _center(H_tot - H_f[f], sB_tr, sB_tr, mB, mB, n_tr)
            H_te = _center(H_f[f], sB_te, sB_te, mB, mB, n_te)
            BY_tr = _center(BY_tot - BY_f[f], sB_tr, xp.broadcast_to(self.sY_tr[f], (K, ch)), mB, mY, n_tr)
            BY_te = _center(BY_f[f], sB_te, xp.broadcast_to(self.sY_te[f], (K, ch)), mB, mY, n_te)
            ss_res = xp.zeros((K, ch))
            for a, c in self.alpha_groups:
                c_d = xp.asarray(c)
                D = 1.0 / (self.lam[f] + a)                                   # (p_A,)
                DG = D[None, :, None] * Gt_tr                                  # (K, p_A, q)
                DY = D[:, None] * self.Yt[f][:, c_d]                           # (p_A, c)
                S = H_tr + a * eye - Gt_tr.transpose(0, 2, 1) @ DG             # (K, q, q)
                rhs = BY_tr[:, :, c_d] - Gt_tr.transpose(0, 2, 1) @ DY         # (K, q, c)
                Wb = xp.linalg.solve(S, rhs)                                   # (K, q, c)
                Wa = DY[None] - DG @ Wb                                        # (K, p_A, c)
                yte = self.yte[f][:, c_d]
                AWa = (self.Ate[f] @ Wa.transpose(1, 0, 2).reshape(p_a, -1)).reshape(p_a, K, -1).transpose(1, 0, 2)
                ss = (self.yy_te[f][c_d][None]
                      - 2 * xp.einsum('kpc,pc->kc', Wa, yte)
                      - 2 * xp.sum(Wb * BY_te[:, :, c_d], axis=1)
                      + xp.sum(Wa * AWa, axis=1)
                      + 2 * xp.sum(Wa * (Gt_te @ Wb), axis=1)
                      + xp.sum(Wb * (H_te @ Wb), axis=1))
                ss_res[:, c_d] = ss
            sst = self.sst_f[f]
            r2 += xp.where(sst[None] == 0, 0.0, 1 - ss_res / xp.where(sst == 0, 1.0, sst)[None])
        r2 /= F
        return r2.get() if self.use_gpu else r2

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
