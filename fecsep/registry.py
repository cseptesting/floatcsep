import os
import h5py
import numpy
from datetime import datetime
from dataclasses import dataclass, field
from csep.utils.time_utils import decimal_year
from functools import wraps


@dataclass()
class ModelTree:
    path: None
    dir: str = None
    fmt: str = None


@dataclass
class Registry:
    _path: str = None
    _class: str = None
    meta: dict = field(default_factory=dict)
    tree: dict = field(default_factory=dict)

    @property
    def name(self):
        return self.meta['name']

    @property
    def path(self):
        if self._path is None:
            return self.meta['path']
        else:
            return self._path

    @path.setter
    def path(self, new_path):
        self._path = new_path

    @property
    def dir(self) -> str:
        """
        Returns:
            The directory containing the model source.
        """
        if os.path.isdir(self.path):
            return self.reg.path
        else:
            return os.path.dirname(self.path)

    @property
    def fmt(self) -> str:
        return os.path.splitext(self.path)[1][1:]

    def add_reg(self, reg):
        self.tree[reg.name] = reg

    def exists(self, tstring, **kwargs):
        if self._class == 'Model':
            return self.forecast_exists(tstring)

    def forecast_exists(self, tstring):
        return tstring


def register(init_func):
    @wraps(init_func)
    def init_with_reg(obj, *args, **kwargs):
        reg = Registry()
        obj.__setattr__('reg', reg)
        init_func(obj, *args, **kwargs)
        reg.meta = obj.to_dict()
        reg._class = obj.__class__.__name__
        print(f'Initialized {obj.name} with reg')

    return init_with_reg
