import os
from dataclasses import dataclass, field
from typing import ClassVar, Union, List, Dict, Protocol, Type


@dataclass()
class DirectoryTree:
    path: str
    dir: str = None
    ext: str = None
    fmt: str = None

    def __getattr__(self, item):
        return self.__dict__[item]


@dataclass
class Registry:
    path: str = os.getcwd()
    tree: dict = field(default_factory=dict)

    def __call__(self, *args, **kwargs):
        print(id(self))

    def add_obj(self, obj):
        if hasattr(obj, 'create_forecast'):  # < Model signature
            ext = os.path.splitext(obj._path)[-1]
            if bool(ext):  # model is a file
                dir_ = os.path.dirname(obj._path)
                fmt = ext.split('.')[-1]
                obj._src = 'file'
            else:  # model is bin
                dir_ = obj.path
                fmt = ''
                obj._src = 'bin'

            self.tree[obj.name] = DirectoryTree(path=obj._path,
                                                dir=dir_,
                                                ext=ext.split('.')[-1],
                                                fmt=fmt)

    def get_path(self, obj):
        if hasattr(obj, 'create_forecast'):  # < Model signature
            return self.tree[obj]['path']


def exp_registry(exp_init):
    def exp_wrapper(*args, **kwargs):
        try:
            path = kwargs.get('path', args[3])
        except IndexError:
            path = os.getcwd()
        args[0].reg = Registry(path)
        exp_init(*args, **kwargs)

    return exp_wrapper


def model_registry(model_init):
    def model_wrapper(*args, **kwargs):
        model_init(*args, **kwargs)

    return model_wrapper
