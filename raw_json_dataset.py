"""Compatibility import; implementation moved unchanged to data/raw_json_dataset.py."""
from data.raw_json_dataset import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("data.raw_json_dataset", run_name="__main__")
