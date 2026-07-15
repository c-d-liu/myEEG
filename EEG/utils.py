# load iEEG data in an object
import os
USE_GPU = os.environ.get("USE_GPU", "0") == "1"

import mne
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
import numpy as np
if USE_GPU:
    from cuml.linear_model import Ridge
    from cuml.metrics import r2_score, mean_squared_error
    print("🟢 Running on GPU (cuML)")
else:
    from sklearn.linear_model import Ridge
    print("🔴 Running on CPU (scikit-learn)")
    from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold
import re
from pprint import pprint

sampf = 120
margin = 2 # seconds
silence = 1.2
onset = margin + silence
samples_per_bin = sampf * 20 # 20 seconds per bin

class EEGData:
    def __init__(self, data: pd.DataFrame, channels: list, sampf: int, margin: int = 2, silence = 1.2, subject = '') -> None:
        self.data = data
        self.channels = channels
        self.sampf = sampf
        self.margin = margin
        self.silence = silence
        self.subject = subject
        self.onset_time = margin + silence
        self.features = []
        self.matrix: 'TRFmatrix|None' = None
    
    # Computed properties
    @property
    def n_channels(self):
        return len(self.channels)

    @property
    def n_samples(self):
        return len(self.data)

    @property
    def duration(self):
        return self.data['time'].max() - self.data['time'].min()

    @property
    def blocks(self):
        return self.data['block'].unique().tolist()

    @property
    def n_blocks(self):
        return len(self.blocks)

    @property
    def bins(self):
        return self.data['bin'].unique().tolist()

    @property
    def n_bins(self):
        return len(self.bins)
    
    def get_eeg(self, channels: list|None = None) -> pd.DataFrame:
        if channels is None:
            channels = self.channels
        return self.data[channels]
    
    def load_features(self, path: str, name: str, zscore = True, col = 'value', onset = 'onset', must_unique = True, existing_feature = 'sum') -> None:
        '''Load features from a csv file and add to the data dataframe. The csv file should have a 'time' column.'''
        if name in self.features:
            existing = True
            if existing_feature == 'sum':
                print(f"Feature {name} already exists, summing the values")
                self.data[name] = self.data[name].add(feat[name], fill_value=0)
            else:
                raise ValueError(f"Feature {name} already exists, choose a different name or set overlap to 'sum'")
        else:
            existing = False
        feat = pd.read_csv(path)
        if zscore:
            feat[col] = (feat[col] - feat[col].mean()) / feat[col].std()
        feat = feat.rename(columns={col: name})
        # convert time to index
        if onset == 'onset':
            feat['time'] += self.onset_time # add onset time to the feature time
        elif onset == 'margin':
            feat['time'] += self.margin # add margin time to the feature time
        elif onset == 'none':
            pass
        else: raise ValueError("onset must be 'onset', 'margin', or 'none'")
        feat['ind'] = (feat['time'] * self.sampf).round().astype(int)
        assert feat['ind'].min() >= 0 and feat['ind'].max() < self.n_samples, f"Feature time is out of bounds: {path}"
        if feat['ind'].duplicated().any():
            if must_unique:
                raise ValueError(f"Feature {name} has duplicated times, set must_unique to False to sum duplicates")
            else:
                # sum duplicates
                print(f"Feature {name} has duplicated times, summing duplicates")
                feat = feat.groupby('ind').sum().reset_index()
        feat = feat.set_index('ind')
        if existing and existing_feature == 'sum':
            self.data[name] = self.data[name].add(feat[name], fill_value=0)
        else:
            self.data = self.data.join(feat[name], how='left').fillna(0)
            self.features.append(name)
    
    def load_preloaded_features(self, feature_series: pd.Series, name: str, existing_feature = 'sum') -> None:
        '''Add a preloaded feature Series (indexed by sample index 'ind') to the data dataframe.'''
        
        # 1. Check for existing features (similar to original logic)
        if name in self.features:
            if existing_feature != 'sum':
                raise ValueError(f"Feature {name} already exists, choose a different name or set existing_feature to 'sum'")
            else:
                # Assuming you still want to handle existing features via summing
                print(f"Feature {name} already exists, summing the preloaded values")
                self.data[name] = self.data[name].add(feature_series, fill_value=0)
        else:
            # 2. Join/Update the main data DataFrame
            # Since 'feature_series' is already indexed by 'ind' and cleaned (no duplicates),
            # we can perform a fast index-aligned join and fill NaNs.
            
            # Merge/join the single column Series.
            temp_df = feature_series.to_frame(name=name)
            
            # Use merge for a clean, explicit join on index
            self.data = self.data.merge(temp_df, 
                                        left_index=True, 
                                        right_index=True, 
                                        how='left')
            
            # Fill NaNs created by the merge (where time indices don't match)
            self.data[name] = self.data[name].fillna(0)
            self.features.append(name)

    def get_matrix(self, cols: None|list = None, window_before: float = 1.0, window_after: float = 0.2) -> 'TRFmatrix':
        '''Get lagged matrix for specified columns'''
        if cols is None:
            cols = self.features
        lagged_list, mask, blocks = make_lagged_matrix(self.data, cols, self.sampf, window_before, window_after)
        return TRFmatrix(lagged_list, mask, blocks, cols, self.sampf, window_before, window_after)
    
    def prepareTRF(self, cols: None|list = None, window_before: float = 1.0, window_after: float = 0.2) -> None:
        '''Prepare TRF matrix and store in self.matrix'''
        self.matrix = self.get_matrix(cols, window_before, window_after)

    @classmethod
    def concat(cls, eeg_list: list['EEGData'], fill_missing: bool = False, fill_value: float = 0.0) -> 'EEGData':
        """Concatenate multiple EEGData objects into a new EEGData instance."""
        if not eeg_list:
            raise ValueError("eeg_list must not be empty")
    
        ref = eeg_list[0]
    
        # Check consistency
        for eeg in eeg_list:
            if eeg.sampf != ref.sampf:
                raise ValueError("All EEGData objects must have the same sampling rate")
            if eeg.channels != ref.channels:
                raise ValueError("All EEGData objects must have the same channel list")
            if set(eeg.features) != set(ref.features):
                if not fill_missing:
                    raise ValueError(f"Feature mismatch between EEG objects: {eeg.features} vs {ref.features}")
        
        # Optionally fill missing features
        if fill_missing:
            all_features = set(sum((e.features for e in eeg_list), []))
            for eeg in eeg_list:
                missing = all_features - set(eeg.features)
                for f in missing:
                    eeg.data[f] = fill_value
                    eeg.features.append(f)
        else:
            all_features = ref.features  # preserve original order (no sorting)
    
        # Adjust bin numbering
        cumulative_max = 0
        data_list = []
        for eeg in eeg_list:
            df = eeg.data.copy()
            df['bin'] += cumulative_max
            cumulative_max = df['bin'].max() + 1
            data_list.append(df)
    
        combined_data = pd.concat(data_list, ignore_index=True)
    
        # Create new EEGData
        combined = cls(
            combined_data,
            channels=ref.channels,
            sampf=ref.sampf,
            margin=ref.margin,
            silence=ref.silence,
        )
    
        # Inherit shared attributes
        combined.features = list(all_features)
        return combined

    def get_y(self, channels = None):
        """Get EEG data for specific channels
        the returned y is aligned with the last sample in window_before
        For example, if window_before=1s, window_after=0.2s, sampf=100Hz,
        the first sample in y corresponds to the 100th sample in the original data,"""
        if self.matrix is None:
            raise ValueError("TRF matrix not prepared. Call prepareTRF() first.")
        if channels is None:
            channels = self.channels
        # Use the mask from the TRF matrix to select valid rows
        mask = self.matrix.mask
        window_before = self.matrix.window_before
        window_after = self.matrix.window_after
        sampf = self.matrix.sampf
        # Extract EEG data for the channels in the TRF matrix
        y_start = int(window_before*sampf-1)  # start of the window in samples, 0-indexed
        y_end = -int(window_after*sampf)  # end of the window in samples
        y = self.get_eeg(channels).iloc[y_start:y_end][mask] # the first y is aligned with the last sample in window_before
        return y, y_start, y_end

    def get_x(self, features: list):
        """Get lagged feature matrix for specific features"""
        if self.matrix is None:
            raise ValueError("TRF matrix not prepared. Call prepareTRF() first.")
        submatrix = self.matrix.subset(features)
        predictors_map = submatrix.predictor_map
        mask = submatrix.mask
        Xs = submatrix.lagged_list
        X = np.hstack([Xs[predictors_map[s]] for s in features])[mask]

        return X

    def fit_ridge_regression(self, features: list, alphas: list[float] = [1000], n_splits: int = 10, random_state=42, optimize_alpha: bool = False) -> "RidgeModel":
        """Fit ridge regression model."""
        if self.matrix is None:
            self.prepareTRF(features)
        y, y_start, y_end = self.get_y()
        submatrix = self.matrix.subset(features)
        predictors_map = submatrix.predictor_map
        Xs = submatrix.lagged_list
        Xshape = [X.shape[1] for X in Xs]
        mask = self.matrix.mask
        blocks = self.data['block'].iloc[y_start:y_end][mask]
        groups = self.data['bin'].iloc[y_start:y_end][mask]
        X = np.hstack([Xs[predictors_map[s]] for s in features])[mask]
        weights, alphas, best_r2 = ridge_cv_stratified_group(
            X, y, blocks, groups,
            alphas=alphas,
            n_splits=n_splits,
            random_state=random_state,
            optimize_alpha=optimize_alpha
        )
        print("Best alpha for the model:", alphas)
        return RidgeModel(weights, self.sampf, alphas, best_r2, Xshape, submatrix.window_before, submatrix.window_after, features=features, channels=self.channels)

    def permute_column(self, column: str) -> None:
        """Permute non-zero values in a column in the data."""
        if column not in self.data.columns:
            raise ValueError(f"Column {column} not found in data.")
        non_zero_mask = self.data[column] != 0
        self.data.loc[non_zero_mask, column] = np.random.permutation(self.data.loc[non_zero_mask, column].values)


def read_fif(path: str, sampf: int = sampf, margin: float|int = margin, silence: float = silence, samples_per_bin: int = samples_per_bin, block: int = 1, exclude = ['bad_interruption'],
             tmax: float|int|None = None) -> EEGData:
    onset = margin + silence
    raw: mne.io.Raw = mne.io.read_raw_fif(path, preload=True)
    first_time = raw.first_time
    if tmax is not None:
        raw.crop(tmax=tmax)
    df = raw.resample(sampf).to_data_frame()
    df.iloc[:,1:] = df.iloc[:,1:].apply(lambda x: (x - x.mean()) / x.std(), axis=0) # z-score the data
    df['block'] = block
    df["bin"] = df.index // samples_per_bin # create a grouping variable for shuffling
    df['onset'] = df['time'] < onset

    # Mark bad intervals
    bads = [anno for anno in raw.annotations if anno['description'].lower().startswith('bad')]
    bad_intervals = [(anno['onset'], anno['onset'] + anno['duration']) for anno in bads]
    bad_intervals = [(start-first_time, end-first_time) for start, end in bad_intervals] # adjust to first_time
    print(f"Found {len(bad_intervals)} bad intervals in {path}")
    pprint("Bad intervals:")
    pprint(bad_intervals)
    if any(end - start <= 1/sampf for start, end in bad_intervals):
        print("Warning: Found bad intervals with duration <= 1/sampf. It might be ignored.")
    def mark_bad(time, bad_intervals=bad_intervals):
        return any(start <= time <= end for start, end in bad_intervals)
    # After optimization:
    time_array = df['time'].values
    bad_mask = np.zeros_like(time_array, dtype=bool)
    exclude_mask = np.zeros_like(time_array, dtype=bool)
    # Vectorized bad/exclude marking (replace loops with np.logical_or)
    # For 'bad' intervals
    for start, end in bad_intervals:
        bad_mask = np.logical_or(bad_mask, (time_array >= start) & (time_array <= end))
    df['bad'] = bad_mask

    # For 'exclude' intervals
    exclude_intervals = []
    for ex in exclude:
        annos = [anno for anno in raw.annotations if anno['description'].lower() == ex.lower()]
        for a in annos:
            print(f"Bad interval for {ex}: onset={a['onset']-first_time}, duration={a['duration']}")
        intervals = [(anno['onset']-first_time, anno['onset'] + anno['duration']-first_time) for anno in annos]
        exclude_intervals.extend(intervals) 
    for start, end in exclude_intervals:
        exclude_mask = np.logical_or(exclude_mask, (time_array >= start) & (time_array <= end))
    
    # DO NOT DROP INDEX. IT CHANGES ALIGNMENT.
    df = df[~exclude_mask] # exclude exclude intervals

    channels = df.columns.tolist()
    print("Channels loaded:", channels)
    channels.remove('time')
    channels.remove('block')
    channels.remove('bin')
    channels.remove('onset')
    channels.remove('bad')
    return EEGData(df, channels, sampf, margin, silence)

def preload_feature_dir(dir, sampf, margin, silence, zscore = False, blocks = [1,2,3,4], sum_duplicate_times = True, sum_duplicate_names = True,
                        pattern: str = r'([a-zA-z]+)\d*_(\d+)\.csv'):
    '''Preload all features from a directory
    Files should be named as 'featureName_blockNum.csv', e.g. wordFreq_1.csv
    returns a dict of features per block: {block_num: {feature_name: pd.Series}}
    '''
    featuref = os.listdir(dir)
    feature_per_block = {i: {} for i in blocks}
    for f in featuref:
        match = re.match(pattern, f)
        if not match:
            raise ValueError(f"Filename {f} does not match pattern")
        name = match.group(1)
        block = int(match.group(2))
        if block not in blocks:
            print(f"Skipping feature {f} for block {block} not in {blocks}")
            continue
        feat = pd.read_csv(os.path.join(dir, f))
        feat['time'] += margin + silence # add onset time to the feature time
        feat['ind'] = (feat['time'] * sampf).round().astype(int)
        if feat['ind'].duplicated().any():
            if not sum_duplicate_times:
                raise ValueError(f"Feature {name} has duplicated times, set sum_duplicate_times to True to sum duplicates")
            else:
                feat = feat.groupby('ind').sum().reset_index()
        feat = feat.set_index('ind')
        if zscore:
            feat['value'] = (feat['value'] - feat['value'].mean()) / feat['value'].std()
        # sum duplicate names
        if name in feature_per_block[block]:
            if not sum_duplicate_names:
                raise ValueError(f"Feature {name} already exists for block {block}, set sum_duplicate_names to True to sum duplicates")
            else:
                print(f"Feature {name} already exists for block {block}, summing the values")
                feature_per_block[block][name] = feature_per_block[block][name].add(feat['value'], fill_value=0)
        else:
            feature_per_block[block][name] = feat['value']
    return feature_per_block

def merge_feature_dicts(d1, d2, sum_duplicate_names=True):
    """
    Merge two feature_per_block dictionaries.

    Parameters
    ----------
    d1, d2 : dict
        {block_num: {feature_name: pd.Series}}
    sum_duplicate_names : bool
        If True, sum Series with the same feature name.
        If False, raise an error on conflicts.

    Returns
    -------
    merged : dict
        Merged feature dictionary.
    """
    merged = {}

    all_blocks = set(d1) | set(d2)

    for block in all_blocks:
        merged[block] = {}

        # Features from both dicts for this block
        f1 = d1.get(block, {})
        f2 = d2.get(block, {})

        all_features = set(f1) | set(f2)

        for name in all_features:
            if name in f1 and name in f2:
                if not sum_duplicate_names:
                    raise ValueError(
                        f"Feature '{name}' already exists for block {block}"
                    )
                merged[block][name] = f1[name].add(f2[name], fill_value=0)
            elif name in f1:
                merged[block][name] = f1[name]
            else:
                merged[block][name] = f2[name]

    return merged

def make_lagged_matrix(df, cols, sampf, window_before=1.0, window_after=0.2, bad_pattern = r'^bad'):
    """
    Create lagged values for specified columns in a dataframe using a sliding window.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the columns.
    cols : list of str
        List of column names to include. 'block' must be included.
        If not the last element, it will be moved to the end.
        Note that 'block' is not lagged but used for masking.
        It is not included in the output list.
    sampf : int
        Sampling frequency.
    window_before : float, optional
        Time window (in seconds) before the current sample.
    window_after : float, optional
        Time window (in seconds) after the current sample.
    bad_pattern : str, optional
        Regex pattern for columns that mark bad segments, which will be excluded, by default r'^bad'

    Returns
    -------
    TRFmatrix
    """

    # Ensure 'block' is included and always last
    cols = [c for c in cols if c != 'block'] + ['block']
    assert pd.Series(cols).is_unique, "Duplicate features exist"
    bad_cols = [c for c in df.columns if re.match(bad_pattern, c)] # columns marking bad segments
    if bad_cols and np.any(df[bad_cols].sum(axis=1) > 0):
        print("Found bad segments in", bad_cols)
        cols += bad_cols

    # Select relevant columns
    dfs = df[cols]

    # Compute window size, ensure consistency with get_y
    window = int(window_before * sampf) + int(window_after * sampf)
    expected_row = dfs.shape[0] - window + 1
    expected_col = dfs.shape[1] * window

    # Sliding window matrix
    matrix = sliding_window_view(dfs, window_shape=window, axis=0).reshape(expected_row, expected_col)

    # Create a mapping for each column to its lagged positions
    new_cols = dfs.columns.repeat(window)
    col_map = {s: new_cols == s for s in dfs.columns}

    # Extract blocks and ensure masking by block boundaries
    blocks = matrix[:, col_map['block']]
    #assert np.all(np.unique(blocks) == np.array([1., 2., 3.])) or np.all(np.unique(blocks) == np.array([1., 2., 3., 4.]))
    mask = np.all(blocks == blocks[:, [0]], axis=1) # keep only rows where block is consistent
    n_samples_crossing = np.sum(~mask)
    print(f"Found {n_samples_crossing} samples crossing block boundaries.")
    if bad_cols:
        for bad_col in bad_cols:
            bad_values = matrix[:, col_map[bad_col]]
            mask = mask & (bad_values.sum(axis=1) == 0)  # exclude rows with any bad values
        n_samples_bad = np.sum(~mask) - n_samples_crossing
        print(f"Found {n_samples_bad} samples with bad values.")

    # Return lagged values in order of `cols`
    lagged_list = [matrix[:, col_map[c]] for c in cols if c != 'block' and c not in bad_cols]

    return lagged_list, mask, blocks

class TRFmatrix:
    def __init__(self, lagged_list: list[np.ndarray], mask: np.ndarray, blocks: np.ndarray, predictors: list[str], sampf: int, window_before, window_after) -> None:
        self.lagged_list = lagged_list
        self.mask = mask
        self.blocks = blocks
        self.predictors = predictors
        self.sampf = sampf
        self.window_before = window_before
        self.window_after = window_after

    @property
    def predictor_map(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.predictors)}
    
    def subset(self, names: list[str]) -> 'TRFmatrix':
        assert isinstance(names, list), "names must be a list of predictor names"
        indices = [self.predictor_map[name] for name in names]
        sublist = [self.lagged_list[i] for i in indices]
        return TRFmatrix(sublist, self.mask, self.blocks, names, self.sampf, self.window_before, self.window_after)
    
    def plot_features(self, start, end, title: None|str = None) -> None:
        '''start: start in seconds
           end: end in seconds'''
        # convert to samples
        if start is not None:
            start_i = int(start * self.sampf)
        else:
            start_i = 0
        if end is not None:
            end_i = int(end * self.sampf)
        else:
            end_i = len(self.mask)
        n = len(self.lagged_list)
        plotlist = [t[start_i:end_i,-1] for t in self.lagged_list.copy()]
        fig, axes = plt.subplots(n, 1, figsize=(10, 2*n), sharex=True)
        if n == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.plot(np.arange(start_i, end_i)/self.sampf, plotlist[i], alpha=0.5)
            ax.set_title(self.predictors[i])
            ax.set_ylabel('zscores')
        axes[-1].set_xlabel('Seconds')
        plt.tight_layout()
        if title is not None:
            plt.title(title)
        plt.show()

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

# Helper function to safely move CuPy arrays to CPU 
# without crashing if they are already NumPy arrays.
def to_cpu(obj):
    if hasattr(obj, 'get'):
        return obj.get()
    return np.asarray(obj)

def ridge_cv_stratified_group(X, y, blocks, groups, alphas, n_splits=5, random_state=None,
                              USE_GPU=USE_GPU, optimize_alpha=False):
    # use this function to pass a single alpha value to just fit a model with fixed alpha,
    # or a list of alpha values to fit the model and find the best alpha per channel
    # it is not possible to pass per channel alpha values
    X = np.asarray(X)
    y = np.asarray(y)
    assert len(X) == len(y) == len(blocks) == len(groups), f"X, y, blocks, and groups must have the same length.\nFound {(len(X), len(y), len(blocks), len(groups))}"
    results = {"alpha": [], "mean_r2": [], "mean_mse": [], "fold_scores": {}, "models": {}, "mean_r2_channels": {}}

    if optimize_alpha:
        for alpha in alphas:
            print(f"Testing alpha: {alpha}")
            r2_scores, mse_scores, fold_models = [], [], []

            for train_idx, test_idx in stratified_group_kfold(blocks, groups,
                                                            n_splits=n_splits,
                                                            random_state=random_state):
                if USE_GPU:
                    model = Ridge(alpha=alpha, solver='lsmr')
                else:
                    model = Ridge(alpha=alpha)
                model.fit(X[train_idx], y[train_idx])
                y_pred = model.predict(X[test_idx])

                r2_scores.append(r2_score(y[test_idx], y_pred, multioutput="raw_values"))
                mse_scores.append(mean_squared_error(y[test_idx], y_pred))
                #pearson = np.corrcoef(y[test_idx].T, y_pred.T).diagonal(offset=y.shape[1])
                fold_models.append(model)

            r2_scores = np.stack(r2_scores, axis=0)   # shape: [n_folds, n_channels]
            mean_r2_channels = np.mean(r2_scores, axis=0)  # shape: [n_channels]

            results["alpha"].append(alpha)
            results["mean_r2"].append(np.mean(mean_r2_channels)) # global average
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
        best_alpha_per_channel = np.array([alphas[i] for i in best_alpha_indices])
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

        results["avg_coefs_best"] = avg_coefs_best  # [n_targets, n_features]


        ## Optionally, compute the mean intercepts
        #avg_intercepts_per_alpha = {
        #    alpha: np.mean([m.intercept_ for m in results["models"][alpha]], axis=0)
        #    for alpha in alphas
        #}
        #avg_intercepts_best = np.array([
        #    avg_intercepts_per_alpha[best_alpha_per_channel[t]][t]
        #    for t in range(n_targets)
        #])
        #results["avg_intercepts_best"] = avg_intercepts_best

        if USE_GPU:
            # store in cpu
            avg_coefs_best = to_cpu(avg_coefs_best)
            best_alpha_per_channel = to_cpu(best_alpha_per_channel)
            best_r2_per_channel = to_cpu(best_r2_per_channel)

        return avg_coefs_best, best_alpha_per_channel, best_r2_per_channel

    else:
        if len(alphas) == 1:
            print(f"Using fixed alpha: {alphas[0]}")
            alphas = alphas * y.shape[1]  # replicate the single alpha for all channels
        else:
            assert len(alphas) == y.shape[1], "If optimize_alpha is False, the length of alphas must match the number of channels in y"
        r2_scores, mse_scores, fold_models = [], [], []
        for train_idx, test_idx in stratified_group_kfold(blocks, groups,
                                                            n_splits=n_splits,
                                                            random_state=random_state):
            if USE_GPU:
                model = Ridge(alpha=alphas, solver='lsmr')
            else:
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

        if USE_GPU:
            avg_coefs_best = to_cpu(avg_coefs_best)
            alphas = to_cpu(alphas)
            mean_r2_channels = to_cpu(mean_r2_channels)
        
        return avg_coefs_best, alphas, mean_r2_channels


def load_batch(eeg_dir, feature_dir, sampf, margin, silence, samples_per_bin, categorical = ['miscOnsets', 'textPredicted', 'predictedWord']):
    '''Legacy function.
    Load a batch of EEG data and corresponding features from directories.
    Concatenate them into a single EEGData object.
    eeg_dir: directory containing .fif files. Should be named as 'subj_block.fif', e.g. KEH001_1.fif
    feature_dir: directory containing feature .csv files
    sampf: sampling frequency
    margin: margin time in seconds
    silence: silence time in seconds
    samples_per_bin: number of samples per bin for grouping
    categorical: list of feature names that are not z-scored'''
    pattern = r'([a-zA-z]+)\d*_\d\.csv'
    featuref = os.listdir(feature_dir)
    eegpattern = r'.+_(\d+)\.fif'
    eegf = [f for f in os.listdir(eeg_dir) if f.endswith('.fif')]
    def extract_num(f):
        m = re.match(eegpattern, f)
        if m:
            return int(m.group(1))
        else:
            raise ValueError(f"Filename {f} does not match pattern")
    eegf = sorted(eegf, key=extract_num)
    eeglist = []
    for i in range(1, len(eegf)+1):
        eeg = read_fif(os.path.join(eeg_dir, eegf[i-1]), sampf, margin, silence, samples_per_bin, block=i)
        for f in featuref:
            if f.endswith(f'{i}.csv'):
                name = re.match(pattern, f).group(1)
                if name in categorical:
                    eeg.load_features(os.path.join(feature_dir, f), name, zscore=False, col='value', onset='onset', must_unique=False)
                else:
                    eeg.load_features(os.path.join(feature_dir, f), name, zscore=True, col='value', onset='onset', must_unique=False)
        eeglist.append(eeg)

    eeg = EEGData.concat(eeglist)

    return eeg

def load_subject(eeg_dir, sampf: int, margin: float, silence: float, samples_per_bin: int, feature_dir: str = 'stimulus_features', categorical_dir: str = 'stimulus_features_categorical',
                 pattern: str = r'([a-zA-z]+)\d*_\d\.csv', eegpattern: str = r'.+_(\d+)\.fif',
                 preloaded: bool = False, preloaded_cont: dict = {},  # {block_num: {name: DataFrame}}
                 preloaded_cat: dict = {},  # {block_num: {name: DataFrame}}
                 blocks = [1,2,3,4], tmax: float|int|dict|None = None):
    '''Load a batch of EEG data and corresponding features from directories.
    Concatenate them into a single EEGData object.
    eeg_dir: directory containing .fif files. Should be named as 'subj_block.fif', e.g. KEH001_1.fif
    feature_dir: directory containing feature .csv files
    categorical_dir: directory containing categorical feature .csv files, will not be z-scored
    sampf: sampling frequency
    margin: margin time in seconds
    silence: silence time in seconds
    samples_per_bin: number of samples per bin for grouping
    pattern: regex pattern to extract feature name from filename
    eegpattern: regex pattern to extract block number from eeg filename
    preloaded: whether to use preloaded features
    preloaded_cont: dict of preloaded continuous features per block
    preloaded_cat: dict of preloaded categorical features per block
    blocks: list of block numbers to load
    '''
    if preloaded:
        featuref = []
    else:
        featuref = os.listdir(feature_dir)
    eegf = [f for f in os.listdir(eeg_dir) if f.endswith('.fif') and (not f.startswith('._'))]  # skip hidden files
    def extract_num(f):
        m = re.match(eegpattern, f)
        if m:
            return int(m.group(1))
        else:
            raise ValueError(f"Filename {f} does not match pattern")
    eeg_dict = {}
    for f in eegf:
        block_num = extract_num(f)
        if block_num in blocks:
            eeg_dict[block_num] = f
        else:
            print(f"Skipping block {block_num} in {f} not in {blocks}")
    eeglist = []
    for i in blocks:
        if isinstance(tmax, dict):
            tmax_block = tmax.get(i, None)
        else:
            tmax_block = tmax
        eeg = read_fif(path = os.path.join(eeg_dir, eeg_dict[i]), sampf=sampf, margin=margin, silence=silence, samples_per_bin=samples_per_bin, block=i, tmax=tmax_block)
        # load continuous features
        if preloaded:
            if i in preloaded_cont:
                for name, feat in preloaded_cont[i].items():
                    eeg.load_preloaded_features(feat, name)
            else:
                print(f"Warning: No preloaded continuous features for block {i}")
        else:
            for f in featuref:
                if f.endswith(f'{i}.csv'):
                    name = re.match(pattern, f).group(1)
                    eeg.load_features(os.path.join(feature_dir, f), name, zscore=True, col='value', onset='onset', must_unique=False)
        # load categorical features
        if preloaded:
            if i in preloaded_cat:
                for name, feat in preloaded_cat[i].items():
                    eeg.load_preloaded_features(feat, name)
            else:
                print(f"Warning: No preloaded categorical features for block {i}")
        else:
            categoricalf = os.listdir(categorical_dir)
            for f in categoricalf:
                if f.endswith(f'{i}.csv'):
                    name = re.match(pattern, f).group(1)
                    eeg.load_features(os.path.join(categorical_dir, f), name, zscore=False, col='value', onset='onset', must_unique=False)
        eeglist.append(eeg)

    eeg = EEGData.concat(eeglist)
    eeg.subject = os.path.basename(eeg_dir)

    return eeg

def load_subject_podcast(eeg_dir, sampf: int, margin: float, silence: float, samples_per_bin: int, feature_dir: str = 'stimulus_features', categorical_dir: str = 'stimulus_features_categorical',
                 pattern: str = r'([a-zA-z]+)\d*_\d\.csv', eegpattern: str = r'.+_(\d+)\.fif',
                 preloaded: bool = False, preloaded_cont: dict = {},  # {block_num: {name: DataFrame}}
                 preloaded_cat: dict = {}  # {block_num: {name: DataFrame}}
                 ):
    '''Load a batch of EEG data and corresponding features from directories.
    Concatenate them into a single EEGData object.
    eeg_dir: directory containing .fif files. Should be named as 'subj_block.fif', e.g. KEH001_1.fif
    feature_dir: directory containing feature .csv files
    categorical_dir: directory containing categorical feature .csv files, will not be z-scored
    sampf: sampling frequency
    margin: margin time in seconds
    silence: silence time in seconds
    samples_per_bin: number of samples per bin for grouping
    pattern: regex pattern to extract feature name from filename
    eegpattern: regex pattern to extract block number from eeg filename
    preloaded: whether to use preloaded features
    preloaded_cont: dict of preloaded continuous features per block
    preloaded_cat: dict of preloaded categorical features per block'''
    eegf = [f for f in os.listdir(eeg_dir) if f.endswith('.fif')]
    eegf = [f for f in eegf if not os.path.islink(os.path.join(eeg_dir, f))]  # skip symlinks
    def extract_num(f):
        return 0
    eegf = sorted(eegf, key=extract_num)
    assert len(eegf) > 0, f"No .fif files found in {eeg_dir}"
    eeglist = []
    for i in range(1, len(eegf)+1):
        eeg = read_fif(path = os.path.join(eeg_dir, eegf[i-1]), sampf=sampf, margin=margin, silence=silence, samples_per_bin=samples_per_bin, block=i)
        # load continuous features
        if preloaded:
            if i in preloaded_cont:
                for name, feat in preloaded_cont[i].items():
                    eeg.load_preloaded_features(feat, name)
            else:
                print(f"Warning: No preloaded continuous features for block {i}")
        else:
            featuref = os.listdir(feature_dir)
            for f in featuref:
                if f.endswith(f'{i}.csv'):
                    name = re.match(pattern, f).group(1)
                    eeg.load_features(os.path.join(feature_dir, f), name, zscore=True, col='value', onset='onset', must_unique=False)
        # load categorical features
        if preloaded:
            if i in preloaded_cat:
                for name, feat in preloaded_cat[i].items():
                    eeg.load_preloaded_features(feat, name)
            else:
                print(f"Warning: No preloaded categorical features for block {i}")
        else:
            categoricalf = os.listdir(categorical_dir)
            for f in categoricalf:
                if f.endswith(f'{i}.csv'):
                    name = re.match(pattern, f).group(1)
                    eeg.load_features(os.path.join(categorical_dir, f), name, zscore=False, col='value', onset='onset', must_unique=False)
        eeglist.append(eeg)

    eeg = EEGData.concat(eeglist)
    eeg.subject = os.path.basename(eeg_dir)

    return eeg

def load_dataset(dataset_dir, sampf: int, margin: float, silence: float, samples_per_bin: int, subjects: list = None, feature_dir: str = 'stimulus_features', categorical_dir: str = 'stimulus_features_categorical', pattern: str = r'([a-zA-z]+)\d*_\d\.csv', eegpattern: str = r'.+_(\d+)\.fif', preloaded: bool = False, preloaded_cont: dict = {}, preloaded_cat: dict = {}):
    '''Load multiple subjects from a dataset directory.
    dataset_dir: directory containing subject subdirectories
    subjects: list of subject subdirectory names to load. If None, load all.
    feature_dir: directory containing feature .csv files, will be ignored if preloaded is True
    categorical_dir: directory containing categorical feature .csv files, will be ignored if preloaded is True
    sampf: sampling frequency
    margin: margin time in seconds
    silence: silence time in seconds
    samples_per_bin: number of samples per bin for grouping
    pattern: regex pattern to extract feature name from filename
    eegpattern: regex pattern to extract block number from eeg filename
    preloaded: whether to use preloaded features
    preloaded_cont: dict of preloaded continuous features per block
    preloaded_cat: dict of preloaded categorical features per block'''
    subject_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if subjects is not None:
        subject_dirs = [d for d in subject_dirs if d in subjects]
    eeglist = []
    for subj in subject_dirs:
        eeg_dir = os.path.join(dataset_dir, subj)
        if preloaded:
            eeg = load_subject(eeg_dir=eeg_dir, sampf=sampf, margin=margin, silence=silence, samples_per_bin=samples_per_bin, feature_dir=feature_dir, categorical_dir=categorical_dir, pattern=pattern, eegpattern=eegpattern, preloaded=True, preloaded_cont=preloaded_cont, preloaded_cat=preloaded_cat)
        else:
            eeg = load_subject(eeg_dir=eeg_dir, sampf=sampf, margin=margin, silence=silence, samples_per_bin=samples_per_bin, feature_dir=feature_dir, categorical_dir=categorical_dir, pattern=pattern, eegpattern=eegpattern)
        eeglist.append(eeg)
    return eeglist

def merge_subjects(eeglist):
    '''Merge a list of EEGData objects horizontally into a single EEGData object.
    EEGData objects must have the same number of samples and the same time vector.
    Channel names are modified to include the subject ID to avoid duplicates.
    Only EEG channels are merged. Features of the first EEGData object are kept.
    eeglist: list of EEGData objects'''
    if len(eeglist) == 0:
        raise ValueError("eeglist is empty")
    elif len(eeglist) == 1:
        return eeglist[0]
    else:
        assert all(np.array_equal(eeg.data['time'], eeglist[0].data['time']) for eeg in eeglist), "All EEGData objects must have the same time vector"
        eegs = eeglist.copy()
        # change channel names to include subject ID
        for eeg in eegs:
            for i, ch in enumerate(eeg.channels):
                eeg.channels[i] = f"{eeg.subject}_{ch}"
                eeg.data.rename(columns={ch: f"{eeg.subject}_{ch}"}, inplace=True)
        # concatenate data horizontally
        merged_eeg = eegs[0]
        for eeg in eegs[1:]:
            merged_data = pd.concat([merged_eeg.data, eeg.data[eeg.channels]], axis=1)
            merged_eeg.data = merged_data
            merged_eeg.channels += eeg.channels
            merged_eeg.subject += f"_{eeg.subject}"

        return merged_eeg

class RidgeModel:
    def __init__(self, weights, sampf, alphas, r2, Xshape: list[int], window_before: int, window_after: int, features: list[str] | None = None, channels: list[str] | None = None,) -> None:
        self.weights = weights
        self.sampf = sampf
        self.alphas = alphas
        self.r2 = r2
        self.window_before = window_before
        self.window_after = window_after
        self.Xs = Xshape
        self.features = features
        self.channels = channels
    
    def plot_mean_coefficients(self, channel_index, predictor_names,
                               channels, best_r2, sampf, save_dir=None):
        """
        Plot mean coefficients across CV folds for each channel and save to folder.
        Annotates mean R^2 on each plot.

        Parameters
        ----------
        predictor_names : list of str
            Names for each predictor, used in legend/labels.
        group_split : tuple of (list, list)
            Indices of predictors to plot in top subplot and bottom subplot.
            Example: ([0,1], [2,3]) → predictors[0,1] on top; predictors[2,3] on bottom.
        sampf : int
            Sampling frequency.
        channels : list of str
            Names of EEG/MEG channels (or similar).
        best_r2 : np.ndarray
            Mean R² per channel.
        save_dir : str
            Directory to save plots.
        """
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # Compute lag sizes per predictor
        n_lags = self.Xs
        # Total number of predictors
        n_features = len(self.Xs)
        window_before = self.window_before
        window_after = self.window_after

        # stack all fold coefs [n_folds, n_channels, n_features]
        #coefs = np.stack([m.coef_ for m in self.model], axis=0)
        #mean_coef = np.mean(coefs, axis=0)  # [n_channels, n_features]
        mean_coef = self.weights  # [n_channels, n_features]

        # Split coefficients per predictor
        coef_splits = []
        start = 0
        for nl in n_lags:
            coef_splits.append(mean_coef[:, start:start+nl])
            start += nl

        # Time lags (same for all predictors here)
        window = int((window_before + window_after) * sampf)
        time_lags = np.linspace(-window_after, window_before, num=window)

        ch_name = channels[channel_index]
        plt.figure(figsize=(10, 4))
        for i in range(n_features):
            plt.plot(time_lags, coef_splits[i][channel_index][::-1], label=predictor_names[i])
        plt.axhline(0, color="k", linestyle="--")
        plt.ylabel("Coeff. weight") 
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
        # Annotate R²
        plt.text(0.05, 0.95, f"Mean $R^2$ = {best_r2[channel_index]:.3f}",
            transform=plt.gca().transAxes,
            verticalalignment="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        plt.tight_layout()
        plt.title(f"TRF Coefficients - Channel {ch_name}")
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"TRF_Channel_{ch_name}.png"), bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    def plot_mean_heatmap(self, include_features = [], sort = True, title = '', save_path=None):
        n_lags = self.Xs
        weights = self.weights  # [n_channels, n_features]
        window_before = self.window_before
        window_after = self.window_after
        sampf = self.sampf
        if self.features is None:
            predictor_names = [f'feature_{i}' for i in range(len(n_lags))]
        else:
            predictor_names = self.features
        assert len(include_features) > 0, "include_features must be a non-empty list of feature names to include in the plot."
        if self.channels is None:
            channels = [f'channel_{i}' for i in range(weights.shape[0])]
        else:
            channels = self.channels
        assert all(nl == n_lags[0] for nl in n_lags), "Currently only predictors with the same number of lags for heatmap plot."
        # Total number of predictors
        n_features = len(n_lags)
        assert n_features == len(predictor_names)
        # Total number of channels
        n_channels = weights.shape[0]
        assert n_channels == len(channels)
        r_squared = self.r2  # [n_channels]
        alphas_per_channel = self.alphas  # [n_channels]

        mean_coef = weights  # [n_channels, n_features]

        # Split coefficients per predictor
        coef_per_predictor = []
        start = 0
        for nl in n_lags:
            coef_per_predictor.append(mean_coef[:, start:start+nl])
            start += nl

        # exclude specified features
        coef_per_predictor = [coef for coef, name in zip(coef_per_predictor, predictor_names) if name in include_features]
        exclude_features = [name for name in predictor_names if name not in include_features]
        n_features = n_features - len(exclude_features)

        # reshape to [n_channels, n_predictors, n_lags]
        coef_matrix = np.stack(coef_per_predictor, axis=1)  # [n_channels, n_predictors, n_lags]
        coef_matrix = coef_matrix[:, :, ::-1]  # Reverse time lag axis for plotting

        # sort channels by R^2
        if sort:
            sorted_indices = np.argsort(r_squared)[::-1]
            coef_matrix = coef_matrix[sorted_indices]
            channels = [channels[i] for i in sorted_indices]
            r_squared = [r_squared[i] for i in sorted_indices]
            if alphas_per_channel is not None:
                alphas_per_channel = [alphas_per_channel[i] for i in sorted_indices]

        assert coef_matrix.shape[0] == mean_coef.shape[0]
        assert coef_matrix.shape[1] == n_features
        assert coef_matrix.shape[2] == n_lags[0]

        # Time lags (same for all predictors here)
        window = int((window_before + window_after) * sampf)
        time_lags = np.linspace(-window_after, window_before, num=window)

        # --- Dynamic Grid Calculation (for a near-square layout) ---
        C = int(np.ceil(np.sqrt(n_channels))) # Columns
        R = int(np.ceil(n_channels / C))      # Rows
        
        # Shared color map maximum (Vmax)
        # This is crucial for comparing plots! Use the max value across ALL data.
        Vmax = np.max(coef_matrix)
        #contrast_factor = 0.7
        #Vmax = Vmax * contrast_factor
        Vmin = -Vmax  # Symmetric around zero

        # --- Create the figure and subplots ---
        fig, axes = plt.subplots(R, C, figsize=(3 * C, 3 * R), 
                                squeeze=False) # squeeze=False ensures axes is 2D
        axes_flat = axes.flatten()

        # --- Loop through each electrode and plot ---
        for i, (e_name, trf_matrix) in enumerate(zip(channels, coef_matrix)):
            ax = axes_flat[i]
            r2 = r_squared[i]
            
            # 1. Plot the heatmap
            im = ax.imshow(
                trf_matrix, 
                aspect='auto',                 # Important to stretch correctly
                cmap='seismic',                   # Use a sequential colormap
                vmin=Vmin, vmax=Vmax,             # Crucial for consistent comparison
                origin='lower',                # Puts feature 0 at the bottom
                extent=[time_lags.min(), time_lags.max(), 0, n_features]
            )

            # 2. Set the Title
            ax.set_title(f'{e_name}: $R^2 = {r2:.2f}$, $α={alphas_per_channel[i]:.0f}$' if alphas_per_channel is not None else f'{e_name}: $R^2 = {r2:.2f}$', 
                        fontsize=12, fontweight='bold')
            
            # 3. Handle Axes Ticks and Labels
            
            # Only set feature labels on the first column (or all if R=1)
            if i % C == 0:
                # Set Y-axis labels (features)
                features_present = [name for name in predictor_names if name in include_features]
                ax.set_yticks(np.arange(n_features) + 0.5) # Center tick
                ax.set_yticklabels(features_present, fontsize=10)
                ax.set_ylabel('Features', fontsize=12) 
            else:
                ax.set_yticks([])  # Remove y-ticks for internal columns

            # Set X-axis labels (time)
            ax.set_xticks(np.arange(time_lags.min(), time_lags.max() + 0.1, 0.2))
            ax.tick_params(axis='x', rotation=0)
            
            # Only set x-label on the bottom row
            if i >= n_channels - C:
                ax.set_xlabel('Time (s)', fontsize=12)
            else:
                ax.set_xticklabels([])

        # --- Clean up and Colorbar ---

        # Hide any unused subplots if n_channels is not a perfect square
        for j in range(n_channels, R * C):
            fig.delaxes(axes_flat[j])

        # Add a single colorbar for the whole figure
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.2, 0.02, 0.6]) # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Model beta', rotation=270, labelpad=15, fontsize=12)

        if title:
            fig.suptitle(title, 
                        fontsize=16,          # Choose a suitable size
                        fontweight='bold', 
                        y=1.02)               # y=1.02 slightly pushes the title up
        plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout to make space for cbar
        if save_path:
            # create directory if not exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()

    def get_all_trfs(self):
        '''Return a dict of TRF matrices for all channels, keyed by channel name.'''
        n_lags = self.Xs
        weights = self.weights  # [n_channels, n_features]
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)  # Ensure 2D shape for single channel
        if self.features is None:
            predictor_names = [f'feature_{i}' for i in range(len(n_lags))]
        else:
            predictor_names = self.features
        # Total number of predictors
        n_features = len(n_lags)
        #assert n_features == len(predictor_names)
        if self.channels is None:
            channels = [f'channel_{i}' for i in range(weights.shape[0])]
        else:
            channels = self.channels
        assert len(channels) == weights.shape[0]

        mean_coef = weights  # [n_channels, n_features]

        # Split coefficients per predictor
        coef_per_predictor = []
        start = 0
        for nl in n_lags:
            coef_per_predictor.append(mean_coef[:, start:start+nl])
            start += nl

        # reshape to [n_channels, n_predictors, n_lags]
        coef_matrix = np.stack(coef_per_predictor, axis=1)  # [n_channels, n_predictors, n_lags]
        coef_matrix = coef_matrix[:, :, ::-1]  # Reverse time lag axis for plotting

        trf_dict = {ch: {} for ch in channels}
        for i, ch in enumerate(channels):
            for j, feat in enumerate(predictor_names):
                trf_dict[ch][feat] = coef_matrix[i, j, :]

        return trf_dict

    def get_time(self):
        times = np.arange(-self.window_after, self.window_before, 1/self.sampf) # time points corresponding to the lagged features
        return times


def plot_r2_comparison(r2_x, r2_y, channels,
                       threshold=0.05,
                       xlabel='Best R² (x)',
                       ylabel='Best R² (y)',
                       title='R² Comparison',
                       save_path=None):
    """
    Compare R² values across channels and annotate significant improvements.

    Parameters
    ----------
    r2_x : list or array
        R² values for condition X (e.g., env only).
    r2_y : list or array
        R² values for condition Y (e.g., env + peakRate).
    channels : list of str
        Channel names, must match length of r2_x and r2_y.
    threshold : float, optional
        Minimum improvement (r2_y - r2_x) to annotate a point (default=0.05).
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    title : str, optional
        Title of the plot.
    save_path : str, optional
        If provided, save the figure to this path instead of showing it.
    """
    
    plt.figure(figsize=(5, 5))
    plt.scatter(r2_x, r2_y, alpha=0.7, color='red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # identity line
    min_val, max_val = min(min(r2_x), min(r2_y)), max(max(r2_x), max(r2_y))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x')

    # annotate channels above threshold
    for x, y, ch in zip(r2_x, r2_y, channels):
        if abs(y - x) > threshold:
            plt.annotate(ch, (x, y), textcoords="offset points",
                         xytext=(5, 5), ha='left', fontsize=8)

    plt.legend()
    plt.title(title)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # close the figure to avoid displaying it
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage of the above functions
    eeg_dir = 'subjects/subject001'
    feature_dir = 'stimulus_features'
    categorical_dir = 'stimulus_features_categorical'
    sampf = 120
    margin = 2
    silence = 1.2
    samples_per_bin = sampf * 20
    all_blocks = [1,2,3,4]
    preloaded_continuous = preload_feature_dir(
        feature_dir, sampf, margin, silence, zscore=True, blocks=all_blocks
    )
    preloaded_categorical = preload_feature_dir(
        categorical_dir, sampf, margin, silence, zscore=False, blocks=all_blocks
    )
    eeg = load_subject(eeg_dir, sampf=sampf, margin=margin, silence=silence, samples_per_bin=samples_per_bin,
                       preloaded=True, preloaded_cont=preloaded_continuous, preloaded_cat=preloaded_categorical,
                       eegpattern=r'.+_part(\d+)\.fif')
    eeg.prepareTRF(cols=eeg.features, window_before=1.0, window_after=0.2)
    basemodel = eeg.fit_ridge_regression(features=['peakRate'], alphas=[1000])
    baser2 = basemodel.r2
    print(f"Baseline R² for 'peakRate': {baser2}")