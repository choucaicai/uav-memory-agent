"""Resolve a self-contained policy package for training and saved checkpoints."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace


def resolve_model(module='model_hiramem_depth', class_name='OpenVLAForActionPrediction'):
    if module != 'model_hiramem_depth':
        raise ValueError('Only model_hiramem_depth is included in this project')
    if not module.isidentifier() or not class_name.isidentifier():
        raise ValueError('model_module must be a top-level package; model_class must be a class name')
    model = getattr(importlib.import_module(f'{module}.modeling_prismatic'), class_name)
    processing = importlib.import_module(f'{module}.processing_prismatic')
    return SimpleNamespace(model=model, config=model.config_class,
                           image_processor=processing.PrismaticImageProcessor,
                           processor=processing.PrismaticProcessor,
                           future_depth=getattr(model, 'requires_future_depth', False),
                           auxiliaries=getattr(model, 'auxiliary_module_names',
                                               ('explore_model', 'trajectory_memory')))


def checkpoint_model(directory):
    config = json.loads((Path(directory) / 'config.json').read_text())
    return (config.get('memory_model_module', 'model_hiramem_depth'),
            config.get('memory_model_class', 'OpenVLAForActionPrediction'))


def stamp_model_config(config, module, class_name, auxiliaries):
    config.memory_model_module = module
    config.memory_model_class = class_name
    config.memory_auxiliary_modules = list(auxiliaries)
