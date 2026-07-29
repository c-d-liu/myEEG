import os
import re
from mne.io import read_raw_fif
from .core import EEGData
from pprint import pprint
import numpy as np
import pandas as pd

sampf = 100
margin = 2 # seconds
silence = 1.2
onset = margin + silence
samples_per_bin = sampf * 20 # 20 seconds per bin


def read_fif(path: str, sampf: int = sampf, margin: float|int = margin, silence: float = silence, samples_per_bin: int = samples_per_bin, block: int = 1, exclude = ['bad_interruption'],
             tmax: float|int|None = None) -> EEGData:
    onset = margin + silence
    raw = read_raw_fif(path, preload=True)
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
                 preloaded_cat: dict = {},  # {block_num: {name: DataFrame}}
                 subdir = None
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
    if subdir is not None:
        eeg_dir = os.path.join(eeg_dir, subdir)
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
