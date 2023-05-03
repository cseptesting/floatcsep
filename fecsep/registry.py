import os
from dataclasses import dataclass, field, asdict
from typing import Sequence
from fecsep.utils import timewindow2str


@dataclass
class ModelTree:
    path: str

    def __call__(self, *args, **kwargs):
        return self.path

    def to_dict(self):
        return asdict(self)

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
    run_folder: str = None
    paths: dict = field(default_factory=dict)

    def __call__(self, *args):
        val = self.paths
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

    def __eq__(self, other):
        return self.workdir == other

    def to_dict(self):
        return asdict(self)

    def abs(self, *paths: Sequence[str]) -> str:
        """ Gets the absolute path of a file, when it was defined relative to
         the experiment working dir."""

        _path = os.path.normpath(
            os.path.abspath(os.path.join(self.workdir, *paths)))
        _dir = os.path.dirname(_path)
        return _path

    def absdir(self, *paths: Sequence[str]) -> str:
        """ Gets the absolute path of a file, when it was defined relative to
         the experiment working dir."""

        _path = os.path.normpath(
            os.path.abspath(os.path.join(self.workdir, *paths)))
        _dir = os.path.dirname(_path)
        return _dir

    def fileexists(self, *args):

        abspath = self.__call__(*args)
        return os.path.exists(abspath)

    def set_pathtree(self,
                     timewindows=None,
                     models=None,
                     tests=None,
                     results_path: str = None,
                     run_name: str = None) -> None:
        """

        Creates the run directory, and reads the file structure inside

        Args:
            timewindows: List of time windows, or representing string.
            models: List of models or model names
            tests: List of tests or test names
            results_path: path to store
            run_name: name of run

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
        # results > test_date > time_window > cats / evals / figures

        run_folder = self.abs(results_path or 'results', run_name or '')

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
            'config': self.abs(run_folder, 'run_config.yml'),
            **{win: {
                'models': {model: {'forecasts': None} for model in models},
                'catalog': os.path.join(dirtree[win]['catalog'],
                                        'catalog.json'),
                'evaluations': {
                    test: {
                        model: os.path.join(dirtree[win]['evaluations'],
                                            f'{test}_{model}.json')
                        for model in models
                    }
                    for test in tests},
                'figures': {
                    **{test: self.abs(dirtree[win]['figures'], f'{test}')
                       for test in tests},
                    **{model: self.abs(dirtree[win]['figures'], f'{model}')
                       for model in models},
                    'catalog': self.abs(dirtree[win]['figures'], 'catalog'),
                    'magnitude_time': self.abs(dirtree[win]['figures'],
                                               'magnitude_time')
                }
            } for win in windows}
        }
        self.paths = target_paths
        self.run_folder = run_folder




