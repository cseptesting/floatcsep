import os
from dataclasses import dataclass, field
from functools import wraps
from collections.abc import Mapping, Sequence
from typing import Union, List, Tuple, Callable
from fecsep.utils import timewindow2str, str2timewindow


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
    _paths: dict = field(default_factory=dict)
    _exists: dict = field(default_factory=dict)

    def __call__(self, *args):

        val = self._paths
        for i in args:
            parsed_arg = self._parse_arg(i)
            val = val[parsed_arg]

        return val

    @staticmethod
    def _parse_arg(arg):

        if isinstance(arg, (list, tuple)):
            return timewindow2str(arg)
        elif isinstance(arg, str):
            return arg
        elif hasattr(arg, 'name'):
            return arg.name
        elif hasattr(arg, '__name__'):
            return arg.__name__
        else:
            raise Exception('Arg is not found')

    def __name__(self):
        return self.workdir

    def __eq__(self, other):
        return self.workdir == other

    def to_dict(self):
        # to be implemented
        return self.workdir

    def abs(self, *paths: Sequence[str]) -> Tuple[str, str]:
        """ Gets the absolute path of a file, when it was defined relative to the
        experiment working dir."""

        _path = os.path.normpath(
            os.path.abspath(os.path.join(self.workdir, *paths)))
        _dir = os.path.dirname(_path)
        return _path

    def absdir(self, *paths: Sequence[str]) -> Tuple[str, str]:
        """ Gets the absolute path of a file, when it was defined relative to the
        experiment working dir."""

        _path = os.path.normpath(
            os.path.abspath(os.path.join(self.workdir, *paths)))
        _dir = os.path.dirname(_path)
        return _dir

    def set_pathtree(self,
                     timewindows=None,
                     models=None,
                     tests=None,
                     results_path: str = None,
                     run_name: str = None) -> None:
        """

        Creates the run directory, and reads the file structure inside

        Args:
            results_path: path to store
            run_name: name of run

        Returns:
            run_folder: Path to the run.
             exists: flag if forecasts, catalogs and test_results if they
             exist already
             target_paths: flag to each element of the gefe (catalog and
             evaluation results)

        """
        from fecsep.utils import timewindow2str
        # grab names for creating directories
        windows = timewindow2str(timewindows)
        models = [i.name for i in models]
        tests = [i.name for i in tests]

        # todo create datetime parser for filenames
        if run_name is None:
            run_name = 'run'
            # todo find better way to name paths
            # run_name = f'run_{datetime.now().date().isoformat()}'

        # Determine required directory structure for run
        # results > test_date > time_window > cats / evals / figures

        run_folder = self.abs(results_path or '', run_name)

        subfolders = ['catalog', 'evaluations', 'figures', 'forecasts']
        dirtree = {
            win: {folder: self.abs(run_folder, win, folder) for
                  folder
                  in subfolders} for win in windows}

        # create directories if they don't exist
        for tw, tw_folder in dirtree.items():
            for _, folder_ in tw_folder.items():
                os.makedirs(folder_, exist_ok=True)

        # Check existing files
        files = {win: {name: list(os.listdir(path)) for name, path in
                       windir.items()} for win, windir in dirtree.items()}

        exists = {win: {
            'forecasts': False,
            # todo Modify for time-dependent, and/or forecast storage
            'catalog': any(file for file in files[win]['catalog']),
            'evaluations': {
                test: {
                    model: any(f'{test}_{model}.json' in file for file in
                               files[win]['evaluations'])
                    for model in models
                }
                for test in tests
            }
        } for win in windows}
        # todo: important in time-dependent, and/or forecast storage
        target_paths = {win: {
            'models': {model: {'forecasts': None} for model in models},
            'catalog': os.path.join(dirtree[win]['catalog'], 'catalog.json'),
            'evaluations': {
                test: {
                    model: os.path.join(dirtree[win]['evaluations'],
                                        f'{test}_{model}.json')
                    for model in models
                }
                for test in tests },
            'figures': {**{test: self.abs(dirtree[win]['figures'], f'{test}')
                           for test in tests},
                        **{model: self.abs(dirtree[win]['figures'], f'{model}')
                                                       for model in models}},
                        'catalog': self.abs(dirtree[win]['figures'],
                                            'catalog.png')
        } for win in windows}

        self._paths = target_paths
        self._exists = exists  # todo perhaps method?


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
