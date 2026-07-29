# CPU backend for ridge_cv_stratified_group.
# Only imported when USE_GPU is False, so scikit-learn is the only dependency pulled in.
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

from .cv_helper import stratified_group_kfold


def fit_stratified_group_cv(X, y, blocks, groups, alphas, n_splits=5, random_state=None,
                             optimize_alpha=False):
    """CPU (scikit-learn) implementation of the stratified-group ridge CV fit.
    Mirrors ridge_cv_backend_gpu.fit_stratified_group_cv exactly in inputs/outputs.
    See ridge_cv_stratified_group in utils.py for the public-facing docstring.
    """
    results = {"alpha": [], "mean_r2": [], "mean_mse": [], "fold_scores": {}, "models": {}, "mean_r2_channels": {}}

    if optimize_alpha:
        for alpha in alphas:
            print(f"Testing alpha: {alpha}")
            r2_scores, mse_scores, fold_models = [], [], []

            for train_idx, test_idx in stratified_group_kfold(blocks, groups,
                                                                n_splits=n_splits,
                                                                random_state=random_state):
                model = Ridge(alpha=alpha)
                model.fit(X[train_idx], y[train_idx])
                y_pred = model.predict(X[test_idx])

                r2_scores.append(r2_score(y[test_idx], y_pred, multioutput="raw_values"))
                mse_scores.append(mean_squared_error(y[test_idx], y_pred))
                fold_models.append(model)

            r2_scores = np.stack(r2_scores, axis=0)   # shape: [n_folds, n_channels]
            mean_r2_channels = np.mean(r2_scores, axis=0)  # shape: [n_channels]

            results["alpha"].append(alpha)
            results["mean_r2"].append(np.mean(mean_r2_channels))  # global average
            results["mean_mse"].append(np.mean(mse_scores))
            results["fold_scores"][alpha] = {
                "r2": r2_scores,   # shape [n_folds, n_channels]
                "mse": mse_scores
            }
            results["models"][alpha] = fold_models  # store all models for this alpha
            results["mean_r2_channels"][alpha] = mean_r2_channels        # per channel

        # ------------------------
        # Select best alpha per channel
        # ------------------------
        all_mean_r2 = np.stack([results["mean_r2_channels"][a] for a in alphas], axis=0)  # [n_alphas, n_targets]
        best_alpha_indices = np.argmax(all_mean_r2, axis=0)
        best_alpha_per_channel = np.array(alphas)[best_alpha_indices]
        best_r2_per_channel = np.max(all_mean_r2, axis=0)

        # ------------------------
        # Compute average model coefficients for each target
        # ------------------------
        # Average coefficients across folds for each alpha
        avg_coefs_per_alpha = {}
        for alpha in alphas:
            fold_coefs = np.stack([m.coef_ for m in results["models"][alpha]], axis=0)  # [n_folds, n_targets, n_features]
            avg_coefs_per_alpha[alpha] = np.mean(fold_coefs, axis=0)  # [n_targets, n_features]

        # Select the coefficient row for each target using its best alpha
        n_targets, n_features = y.shape[1], X.shape[1]
        avg_coefs_best = np.zeros((n_targets, n_features))
        for t in range(n_targets):
            alpha_t = best_alpha_per_channel[t]
            avg_coefs_best[t] = avg_coefs_per_alpha[alpha_t][t]

        return avg_coefs_best, best_alpha_per_channel, best_r2_per_channel

    else:
        if len(alphas) == 1:
            print(f"Using fixed alpha: {alphas[0]}")
            alphas = alphas * y.shape[1]  # replicate the single alpha for all channels
            alphas = np.asarray(alphas)
        else:
            assert len(alphas) == y.shape[1], "If optimize_alpha is False, the length of alphas must match the number of channels in y"
            alphas = np.asarray(alphas)
        r2_scores, mse_scores, fold_models = [], [], []
        for train_idx, test_idx in stratified_group_kfold(blocks, groups,
                                                            n_splits=n_splits,
                                                            random_state=random_state):
            model = Ridge(alpha=alphas)
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            r2_scores.append(r2_score(y[test_idx], y_pred, multioutput="raw_values"))
            mse_scores.append(mean_squared_error(y[test_idx], y_pred))
            fold_models.append(model)

        r2_scores = np.stack(r2_scores, axis=0)   # shape: [n_folds, n_channels]
        mean_r2_channels = np.mean(r2_scores, axis=0)  # shape: [n_channels]
        avg_coefs_best = np.mean(np.stack([m.coef_ for m in fold_models], axis=0), axis=0)  # shape: [n_channels, n_features]

        best_alpha_per_channel = alphas

        return avg_coefs_best, best_alpha_per_channel, mean_r2_channels