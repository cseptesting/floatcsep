import dataclasses
import os
from os.path import join, abspath, relpath, normpath, dirname, exists
from dataclasses import dataclass, field
from typing import Sequence
from floatcsep.utils import timewindow2str


@dataclass
class ModelTree:
    workdir: str
    path: str
    database: str = None
    args_file: str = None
    input_cat: str = None
    forecasts: dict = field(default_factory=dict)
    inventory: dict = field(default_factory=dict)

    def __call__(self, *args):

        val = self.__dict__
        for i in args:
            parsed_arg = self._parse_arg(i)
            val = val[parsed_arg]

        return self.abs(val)

    @property
    def fmt(self) -> str:
        if self.database:
            return os.path.splitext(self.database)[1][1:]
        else:
            return os.path.splitext(self.path)[1][1:]

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

    def __eq__(self, other):
        return self.path == other

    def as_dict(self):
        return self.path

    def asdict(self):
        return dataclasses.asdict(self)

    def abs(self, *paths: Sequence[str]) -> str:
        """ Gets the absolute path of a file, when it was defined relative to
         the experiment working dir."""

        _path = normpath(abspath(join(self.workdir, *paths)))
        _dir = dirname(_path)
        return _path

    def absdir(self, *paths: Sequence[str]) -> str:
        """ Gets the absolute path of a file, when it was defined relative to
         the experiment working dir."""

        _path = normpath(abspath(join(self.workdir, *paths)))
        _dir = dirname(_path)
        return _dir

    def fileexists(self, *args):
        file_abspath = self.__call__(*args)
        return exists(file_abspath)

    def build_tree(self,
                   timewindows=None,
                   model_class='ti',
                   prefix=None,
                   args_file=None,
                   input_cat=None) -> None:
        """

        Creates the run directory, and reads the file structure inside

        Args:
            timewindows (list(str)): List of time windows or strings.
            model_class (str): Time-indendent (ti) or time-dependent (td)
            prefix (str): prefix of the model forecast filenames if TD
            args_file (str): input arguments path of the model if TD
            input_cat (str): input catalog path of the model if TD

        Returns:
            run_folder: Path to the run.
             exists: flag if forecasts, catalogs and test_results if they
             exist already
             target_paths: flag to each element of the gefe (catalog and
             evaluation results)

        """
        if timewindows is None:
            return
        windows = timewindow2str(timewindows)
        if model_class == 'ti':
            fname = self.database if self.database else self.path
            fc_files = {win: fname for win in windows}
            fc_exists = {win: exists(fc_files[win]) for win in windows}

        elif model_class == 'td':
            args = args_file if args_file else join('input', 'args.txt')
            self.args_file = join(self.path, args)
            input_cat = input_cat if input_cat else join('input',
                                                         'catalog.csv')
            self.input_cat = join(self.path, input_cat)
            # grab names for creating directories
            subfolders = ['input', 'forecasts']
            dirtree = {folder: self.abs(self.path, folder) for
                       folder in subfolders}

            # create directories if they don't exist
            for _, folder_ in dirtree.items():
                os.makedirs(folder_, exist_ok=True)

            # set forecast names
            fc_files = {win: join(dirtree['forecasts'],
                                  f'{prefix}_{win}.csv')  # todo other formats?
                        for win in windows}

            fc_exists = {win: any(file for file in
                               list(os.listdir(dirtree['forecasts'])))
                      for win in windows}

        self.forecasts = fc_files
        self.inventory = fc_exists


@dataclass
class PathTree:
    workdir: str
    rundir: str = 'results'
    paths: dict = field(default_factory=dict)
    result_exists: dict = field(default_factory=dict)

    def __call__(self, *args):
        val = self.paths
        for i in args:
            parsed_arg = self._parse_arg(i)
            val = val[parsed_arg]
        return self.abs(self.rundir, val)

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

    def __eq__(self, other):
        return self.workdir == other

    def as_dict(self):
        return self.workdir

    def asdict(self):
        return dataclasses.asdict(self)

    def abs(self, *paths: Sequence[str]) -> str:
        """ Gets the absolute path of a file, when it was defined relative to
         the experiment working dir."""

        _path = normpath(abspath(join(self.workdir, *paths)))
        return _path

    def rel(self, *paths: Sequence[str]) -> str:
        """ Gets the relative path of a file, when it was defined relative to
         the experiment working dir."""

        _abspath = normpath(
            abspath(join(self.workdir, *paths)))
        _relpath = relpath(_abspath, self.workdir)
        return _relpath

    def absdir(self, *paths: Sequence[str]) -> str:
        """ Gets the absolute path of a file, when it was defined relative to
         the experiment working dir."""

        _path = normpath(
            abspath(join(self.workdir, *paths)))
        _dir = dirname(_path)
        return _dir

    def reldir(self, *paths: Sequence[str]) -> str:
        """ Gets the absolute path of a file, when it was defined relative to
         the experiment working dir."""

        _path = normpath(
            abspath(join(self.workdir, *paths)))
        _dir = dirname(_path)
        _reldir = relpath(_dir, self.workdir)
        return _reldir

    def fileexists(self, *args):

        file_abspath = self.__call__(*args)
        return exists(file_abspath)

    def build(self,
              timewindows=None,
              models=None,
              tests=None) -> None:
        """

        Creates the run directory, and reads the file structure inside

        Args:
            timewindows: List of time windows, or representing string.
            models: List of models or model names
            tests: List of tests or test names

        Returns:
            run_folder: Path to the run.
             exists: flag if forecasts, catalogs and test_results if they
             exist already
             target_paths: flag to each element of the gefe (catalog and
             evaluation results)

        """
        # grab names for creating directories
        windows = timewindow2str(timewindows)
        models = [i.name for i in models]
        tests = [i.name for i in tests]

        # todo create datetime parser for filenames
        # todo find better way to name paths

        # Determine required directory structure for run
        # results > time_window > cats / evals / figures

        run_folder = self.rundir

        subfolders = ['catalog', 'evaluations', 'figures', 'forecasts']
        dirtree = {
            win: {folder: self.abs(run_folder, win, folder)
                  for folder in subfolders} for win in windows}

        # create directories if they don't exist
        for tw, tw_folder in dirtree.items():
            for _, folder_ in tw_folder.items():
                os.makedirs(folder_, exist_ok=True)

        # Check existing files
        files = {win: {name: list(os.listdir(path)) for name, path in
                       windir.items()} for win, windir in dirtree.items()}

        file_exists = {win: {
            'forecasts': False,
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

        target_paths = {
            'config': 'repr_config.yml',
            'catalog_figure': 'catalog',
            'magnitude_time': 'events',
            **{win: {
                'catalog': join(win, 'catalog', 'test_catalog.json'),
                'evaluations': {
                    test: {
                        model: join(win, 'evaluations',
                                    f'{test}_{model}.json')
                        for model in models
                    }
                    for test in tests},
                'figures': {
                    **{test: join(win, 'figures', f'{test}')
                       for test in tests},
                    'catalog': join(win, 'figures', 'catalog'),
                    'magnitude_time': join(win, 'figures',
                                           'magnitude_time')
                },
                'forecasts': {model: join(win, 'forecasts', f'{model}')
                              for model in models}
            } for win in windows}
        }
        self.paths = target_paths
        self.result_exists = file_exists

