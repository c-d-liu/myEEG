# Codes for TRF analysis

You need to install `mne` and `pandas` to run the script.

Navigate to this directory and install the package with `pip install -e .` 

Use with `from EEG import *` or `import EEG`

To use GPU acceleration with cuda, use

`USE_GPU=1 python my_script.py`

when you call your script in the command line.

If you can't use the way above, you can do

```
# my_script.py
import os

# 1. Set the flag FIRST
os.environ["USE_GPU"] = "1"

# 2. Import your script SECOND
import EEG
```
