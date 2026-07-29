import numpy as np

def fast_ridge_rsquared(X_train, Y_train, X_val, Y_val, alphas):
    """
    Calculates R-squared for Ridge Regression on a validation set across multiple alphas
    without calculating the explicit weight vectors.

    Args:
        X_train (np.ndarray): Training features, shape (N_train, num_features)
        Y_train (np.ndarray): Training targets, shape (N_train, num_channels)
        X_val (np.ndarray): Validation features, shape (N_val, num_features)
        Y_val (np.ndarray): Validation targets, shape (N_val, num_channels)
        alphas (list or np.ndarray): Regularization parameters to test

    Returns:
        dict: A mapping of alpha values to their R-squared arrays (one R2 per channel)
    """
    if Y_train.ndim == 1: Y_train = Y_train[:, np.newaxis]
    if Y_val.ndim == 1: Y_val = Y_val[:, np.newaxis]

    # 1. Covariance matrix of training features
    covmat = X_train.T @ X_train

    # 2. Eigendecomposition (symmetric -> eigh, faster & more stable than eig)
    eigvals, U = np.linalg.eigh(covmat)

    # 3. Precalculate projections (independent of alpha)
    Usr = U.T @ (X_train.T @ Y_train)
    Usl = X_val @ U

    # Precalculate Total Sum of Squares (SS_tot) for R-squared
    SS_tot = np.sum((Y_val - np.mean(Y_val, axis=0)) ** 2, axis=0)

    results = {}
    for alpha in alphas:
        # 4. Tikhonov regularization on eigenvalues
        D = 1.0 / (eigvals + alpha)

        # 5. Fast prediction via broadcasting (replaces explicit weight computation)
        Usl_D = Usl * D
        predY = Usl_D @ Usr

        # 6. R-squared for this alpha
        SS_res = np.sum((Y_val - predY) ** 2, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            r2 = np.where(SS_tot == 0, 0.0, 1 - (SS_res / SS_tot))

        results[alpha] = r2

    return results


def optimal_alpha_rsquared(X_train, Y_train, X_val, Y_val, best_alphas):
    """
    Calculates R-squared directly using a specific optimal alpha for each target channel,
    without looping through channels or calculating the original feature weights.
    
    Args:
        X_train (np.ndarray): Training features, shape (N_train, num_features)
        Y_train (np.ndarray): Training targets, shape (N_train, num_channels)
        X_val (np.ndarray): Validation features, shape (N_val, num_features)
        Y_val (np.ndarray): Validation targets, shape (N_val, num_channels)
        best_alphas (np.ndarray): 1D array of length (num_channels,) with the optimal alpha per channel
        
    Returns:
        np.ndarray: R-squared values for each channel, shape (num_channels,)
    """
    # 1. Eigendecomposition of the covariance matrix
    covmat = X_train.T @ X_train
    eigvals, U = np.linalg.eigh(covmat)
    
    # 2. Precalculate projections
    Usr = U.T @ (X_train.T @ Y_train)  # Shape: (num_features, num_channels)
    Usl = X_val @ U                    # Shape: (N_val, num_features)
    
    # 3. Vectorized Regularization Matrix (The Magic Step)
    # eigvals[:, None] makes it a column vector: (num_features, 1)
    # best_alphas[None, :] makes it a row vector: (1, num_channels)
    # Adding them creates a 2D array of shape (num_features, num_channels)
    D_2d = 1.0 / (eigvals[:, None] + best_alphas[None, :])
    
    # 4. Fast Prediction
    # Multiply D_2d and Usr element-wise (they share the same shape)
    # Then matrix-multiply with Usl to get predictions for all channels at once
    predY = Usl @ (D_2d * Usr)
    
    # 5. Calculate R-squared
    SS_res = np.sum((Y_val - predY)**2, axis=0)
    SS_tot = np.sum((Y_val - np.mean(Y_val, axis=0))**2, axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = np.where(SS_tot == 0, 0.0, 1 - (SS_res / SS_tot))
        
    return r2


def find_optimal_alphas_vectorized(X_train, Y_train, X_val, Y_val, alphas):
    """
    Evaluates all alphas across all channels simultaneously using 3D tensor math.
    Returns the full R2 grid, the optimal alpha per channel, and the best R2 per channel.
    """
    alphas = np.asarray(alphas)
    
    # 1. Eigendecomposition
    covmat = X_train.T @ X_train
    eigvals, U = np.linalg.eigh(covmat)
    
    # 2. Projections
    Usr = U.T @ (X_train.T @ Y_train)  # Shape: (features, channels)
    Usl = X_val @ U                    # Shape: (val_samples, features)
    
    # 3. 2D Regularization Matrix (All features across all alphas)
    # eigvals[:, None] is (features, 1) and alphas[None, :] is (1, alphas)
    D = 1.0 / (eigvals[:, None] + alphas[None, :])  # Shape: (features, alphas)
    
    # 4. 3D Prediction Tensor (The Magic Step)
    # Broadcast Usr (features, channels, 1) and D (features, 1, alphas)
    # This creates a 3D tensor of scaled weights: (features, channels, alphas)
    D_Usr = Usr[:, :, None] * D[:, None, :]
    
    # np.tensordot sums over the 'features' axis.
    # Usl (val_samples, features) dot D_Usr (features, channels, alphas) 
    # Resulting shape: (val_samples, channels, alphas)
    predY_3d = np.tensordot(Usl, D_Usr, axes=([1], [0]))
    
    # 5. Vectorized R-squared across the 3D tensor
    # Expand Y_val to (val_samples, channels, 1) to subtract the 3D predY_3d tensor
    SS_res = np.sum((Y_val[:, :, None] - predY_3d)**2, axis=0) # Shape: (channels, alphas)
    SS_tot = np.sum((Y_val - np.mean(Y_val, axis=0))**2, axis=0) # Shape: (channels,)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Expand SS_tot to (channels, 1) to divide properly
        r2_grid = np.where(SS_tot[:, None] == 0, 0.0, 1 - (SS_res / SS_tot[:, None]))
        
    # 6. Extract the winners instantly
    best_alpha_indices = np.argmax(r2_grid, axis=1)
    best_alphas = alphas[best_alpha_indices]
    
    # Use advanced indexing to pluck out the highest R2 scores
    best_r2_scores = r2_grid[np.arange(len(best_alpha_indices)), best_alpha_indices]

    result = {
        "alphas": alphas,
        "r2_grid": r2_grid,
        "best_alphas": best_alphas,
        "best_r2_scores": best_r2_scores
    }
    
    return result