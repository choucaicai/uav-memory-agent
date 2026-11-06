"""Load machine-specific paths without importing the model or simulator stack."""
import argparse
import os
from pathlib import Path
import yaml


def load_runtime_config(config_path=None):
    selected = config_path or os.environ.get('UAV_MEMORY_CONFIG')
    path = Path(selected).expanduser() if selected else Path(__file__).resolve().parent / 'configs/runtime.yaml'
    path = path.resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f'Expected a YAML mapping in {path}')

    def resolve(value, base):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'Expected a non-empty path in {path}')
        candidate = Path(os.path.expandvars(value)).expanduser()
        return str((candidate if candidate.is_absolute() else base / candidate).resolve())

    for key in ('env_root', 'airsim_config_dir', 'eval_dataset', 'model_path', 'output_root'):
        if key not in config:
            raise ValueError(f'Missing {key} in {path}')
        config[key] = resolve(config[key], path.parent)
    environments = config.get('environments')
    if not isinstance(environments, dict) or not environments:
        raise ValueError(f'Expected a non-empty environments mapping in {path}')
    config['environments'] = {
        name: resolve(script, Path(config['env_root'])) for name, script in environments.items()
    }
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--get', choices=('env_root', 'airsim_config_dir', 'eval_dataset', 'model_path', 'output_root'))
    arguments = parser.parse_args()
    settings = load_runtime_config(arguments.config)
    print(settings[arguments.get] if arguments.get else yaml.safe_dump(settings, sort_keys=False), end='\n' if arguments.get else '')
