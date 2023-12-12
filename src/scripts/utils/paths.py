"""
Exposes common paths useful for manipulating datasets and generating figures.

"""
from pathlib import Path

# Absolute path to the top level of the repository
root = Path(__file__).resolve().parents[3].absolute()

# Absolute path to the `src` folder
src = root / "src"
srcpath = str(src) + "/"

# Absolute path to the `src/data` folder (contains datasets)
data = src / "data"
datapath = str(data) + "/"

# Absolute path to the `src/scripts` folder (contains figure/pipeline scripts)
scripts = src / "scripts"
scriptspath = str(scripts) + "/"