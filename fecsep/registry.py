import os
from dataclasses import dataclass, field
from functools import wraps
from collections.abc import Mapping, Sequence
from typing import Union, List, Tuple, Callable


@dataclass
class ModelTree:
    path: str
    _path: str = None

    def __call__(self, *args, **kwargs):
        return self.path

    @property
    def dir(self) -> str:
        """
        Returns:
            The directory containing the model source.
        """
        if os.path.isdir(self.path):
            return self.path
        else:
            return os.path.dirname(self.path)

    @property
    def fmt(self) -> str:
        return os.path.splitext(self.path)[1][1:]


@dataclass
class PathTree:
    workdir: str

    def __call__(self, *args, **kwargs):
        return self.workdir

    def to_dict(self):
        # to be implemented
        return 'Path Tree'

    @property
    def dir(self) -> str:
        """
        Returns:
            The directory containing the model source.
        """
        if os.path.isdir(self.path):
            return self.path
        else:
            return os.path.dirname(self.path)

    def abs(self, *paths: Sequence[str]) -> Tuple[str, str]:
        """ Gets the absolute path of a file, when it was defined relative to the
        experiment working dir."""

        _path = os.path.normpath(
            os.path.abspath(os.path.join(self.workdir, *paths)))
        _dir = os.path.dirname(_path)
        return _dir, _path


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
            return self.path
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
