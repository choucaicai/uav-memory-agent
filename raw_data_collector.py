"""Compatibility import; implementation moved unchanged to data/raw_data_collector.py."""
from data.raw_data_collector import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("data.raw_data_collector", run_name="__main__")
