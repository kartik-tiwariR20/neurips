"""
loaders/registry.py
--------------------
The ONE place you touch when adding a new dataset's loader. Everything
else (train.py, sweep.py, analyze.py) just asks this registry for
make_datasets(), never imports a specific dataset's loader directly.

To add dataset #3: write loaders/<name>.py with a make_datasets()
function following the same signature as ecg5000.py / har.py, then add
one line here.
"""
from . import ecg5000
from . import har

REGISTRY = {
    "ecg5000": ecg5000,
    "har": har,
}


def get_loader(name):
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(REGISTRY.keys())}"
        )
    return REGISTRY[name]
