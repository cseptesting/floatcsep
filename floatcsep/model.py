import json
import os
import numpy
import csep
import git
import subprocess
import logging

from typing import List, Callable, Union, Dict, Mapping, Sequence
from datetime import datetime

from csep.core.forecasts import GriddedForecast, CatalogForecast
from csep.utils.time_utils import decimal_year

from floatcsep.accessors import from_zenodo, from_git
from floatcsep.readers import ForecastParsers, HDF5Serializer, check_format
from floatcsep.utils import timewindow2str, str2timewindow
from floatcsep.registry import ModelTree

log = logging.getLogger(__name__)


class Model:
    """

    Class defining a forecast generating Model. Initializes a model source
    either from filesystem or web repositories, contains information about
    the Model's typology, and maps accordingly to a forecast generating
    function.

    Args:
        name (str): Name of the model
        path (str): Relative path of the model (file or runnable code)
                    to the Experiment's instance path
        forecast_unit (float): Temporal unit of the forecast. Default to
                               years in time-independent models and days
                               in time-dependent
        use_db (bool): Flag the use of a database for speed/memory purposes
        func (str, ~collections.abc.Callable): Forecast generating function
                                               (for code models)
        func_kwargs (dict): Arguments to pass into `func`
        zenodo_id (int): Zenodo ID or record of the Model
        giturl (str): Link to a git repository
        repo_hash (str): Specific commit/branch/tag hash.
        authors (list[str]): Authors' names metadata
        doi: Digital Object Identifier metadata:

    """

    '''
    Model characteristics:
        Forecast:   - Gridded
                    - Catalog
        Updating:   - Time-Independent
                    - Time-Dependent
        Source:     - File
                    - Code
    Model typology:
    
    To implement in beta version:       
        - (grid - ti - file): e.g. CSEP1 style gridded forecasts
    To implement in further versions:
        - (grid - ti - code):  e.g. smoothed-seismicity model
        - (grid - td - code): e.g. EEPAS, STEP, Italy-NZ OEF models
        - (cat - td - code): e.g. ETAS model code
        - (cat - td - file): e.g OEF-ready Forecasts
        - (grid - td - file): e.g OEF-ready Forecasts

    Get forecasts options:
        - FILE   - read from file, scale in runtime             
                 - drop to db, scale from function in runtime   
        - CODE  - run, read from file              
                - run, store in db, read from db   

    '''

    def __init__(self, name: str, path: str,
                 forecast_unit: float = 1, use_db: bool = False,
                 func: Union[str, Callable] = None, func_kwargs: dict = None,
                 zenodo_id: int = None, giturl: str = None,
                 repo_hash: str = None, authors: List[str] = None,
                 doi: str = None, **kwargs) -> None:

        # todo:
        #  - Instantiate from source code

        # Instantiate attributes
        self.name = name
        self.zenodo_id = zenodo_id
        self.giturl = giturl
        self.repo_hash = repo_hash
        self.authors = authors
        self.doi = doi
        self.forecast_unit = forecast_unit
        self.func = func
        self.func_kwargs = func_kwargs or {}
        self.use_db = use_db

        self.path = path
        args = kwargs.get('args_file', 'args.txt')
        self.tree = ModelTree(self.name, path, args=args)

        # Set model temporal class
        if self.func:
            # Time-Dependent
            self.model_class = 'td'
            self.build = kwargs.get('build', 'docker')
            self.run_prefix = ''

        else:
            # Time-Independent
            self.model_class = kwargs.get('model_class', 'ti')

        # Instantiate attributes to be filled in run-time
        self.forecasts = {}
        self.__dict__.update(**kwargs)

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

    def stage(self, timewindows=None) -> None:
        """
        Pre-steps to make the model runnable before integrating
            - Get from filesystem, Zenodo or Git
            - Pre-check model fileformat
            - Initialize database
            - Run model quality assurance (unit tests, runnable from floatcsep)
        """
        kwargs = {}
        self.get_source(self.zenodo_id, self.giturl, branch=self.repo_hash)
        if self.model_class == 'td':
            self.build_model()
        if self.use_db:
            self.init_db()
        check_format(self.path, self.fmt, self.func)
        self.model_qa()
        if self.model_class == 'td':
            prefix = self.__dict__.get('prefix', None)
            self.tree.set_pathtree(timewindows, prefix=prefix)

    def build_model(self):

        if self.build == 'pip' or self.build == 'venv':
            self.build_venv()

    def build_venv(self):

        venv = os.path.join(self.path, self.__dict__.get('venv', 'venv'))
        venvact = os.path.join(venv, 'bin', 'activate')

        if not os.path.exists(venv):
            log.info(f'Building model {self.name} using pip')
            subprocess.run(['python', '-m', 'venv', venv])
            log.info(f'\tVirtual environment created in {venv}')

            build_cmd = f'source {venvact} &&' \
                        f'pip install --upgrade pip &&' \
                        f'pip install -e {self.path}'

            cmd = ['bash', '-c', build_cmd]

            process = subprocess.Popen(cmd,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       universal_newlines=True)
            for line in process.stdout:
                print(f'\t{line}', end='')
            process.wait()
            print(f'Nested environments is not well supported. '
                  f'Consider using docker instead')

        self.run_prefix = f'cd {self.path} && source {venvact} &&'

    def get_source(self, zenodo_id: int = None, giturl: str = None,
                   force: bool = False,
                   **kwargs) -> None:
        """

        Search, download or clone the model source in the filesystem, zenodo
        and git, respectively. Identifies if the instance path points to a file
        or to its parent directory

        Args:
            zenodo_id (int): Zenodo identifier of the repository. Usually as
             `https://zenodo.org/record/{zenodo_id}`
            giturl (str): git remote repository URL from which to clone the
             source
            force (bool): Forces to re-query the model from a web repository
            **kwargs: see :func:`~floatcsep.utils.from_zenodo` and
             :func:`~floatcsep.utils.from_git`


        """

        if os.path.exists(self.path) and not force:
            return

        os.makedirs(self.dir, exist_ok=True)

        if zenodo_id:
            try:
                from_zenodo(zenodo_id, self.dir if self.fmt else self.path,
                            force=force)
            except (KeyError, TypeError) as msg:
                raise KeyError(f'Zenodo identifier is not valid: {msg}')

        elif giturl:
            try:
                from_git(giturl, self.dir if self.fmt else self.path,
                         **kwargs)
            except (git.NoSuchPathError, git.CommandError) as msg:
                raise git.NoSuchPathError(f'git url was not found {msg}')
        else:
            raise FileNotFoundError('Model has no path or identified')

        if not os.path.exists(self.dir) or not os.path.exists(self.path):
            raise FileNotFoundError(
                f"Directory '{self.dir}' or file {self.path}' do not exist. "
                f"Please check the specified 'path' matches the repo "
                f"structure")

    def init_db(self, dbpath: str = '', force: bool = False) -> None:
        """
        Initializes the database if `use_db` is True. If the model source is a
        file, seralizes the forecast into a HDF5 file. If source is a
        generating function or code, creates an empty DB

        Args:
            dbpath (str): Path to drop the HDF5 database. Defaults to same path
             replaced with an `hdf5` extension
            force (bool): Forces the serialization even if the DB already
             exists

        """
        # todo Think about distinction btwn 'TI' and 'Gridded' models.
        if self.fmt and self.model_class == 'ti':

            parser = getattr(ForecastParsers, self.fmt)
            rates, region, mag = parser(self.path)

            db_func = HDF5Serializer.grid2hdf5
            if not dbpath:
                dbpath = self.path.replace(self.fmt, 'hdf5')

            if not os.path.isfile(dbpath) or force:
                # Drop Source file into DB
                db_func(rates, region, mag,
                        hdf5_filename=dbpath,
                        unit=self.forecast_unit)

            self.path = dbpath

        else:
            raise NotImplementedError('TD serialization not implemented')

    def rm_db(self) -> None:
        """ Clean up the generated HDF5 File"""

        if self.use_db:
            if os.path.isfile(self.path) and self.fmt == 'hdf5':
                os.remove(self.path)
            else:
                print("The HDF5 file does not exist")

    def model_qa(self) -> None:
        """ Run model quality assurance (Unit and Integration tests).
        Should not run if the repo has not been modified.
        Not implemented """
        pass

    def get_forecast(self,
                     tstring: Union[str, list] = None,
                     region=None
                     ) -> Union[GriddedForecast, CatalogForecast,
                                List[GriddedForecast], List[CatalogForecast]]:

        """ Wrapper that just returns a forecast, which should hide the
         access method (db storage, ti_td, etc.) under the hood"""

        if self.model_class == 'ti':

            if isinstance(tstring, str):
                # If only one timewindow string is passed
                try:
                    # If they are retrieved from the Evaluation class
                    return self.forecasts[tstring]
                except KeyError:
                    # In case they are called from postprocess
                    self.create_forecast(tstring)
                    return self.forecasts[tstring]
            else:
                # If multiple timewindow strings are passed
                forecasts = []
                for tw in tstring:
                    if tw in self.forecasts.keys():
                        forecasts.append(self.forecasts[tw])
                if not forecasts:
                    raise KeyError(
                        f'Forecasts {*tstring,} have not been created yet')
                return forecasts

        elif self.model_class == 'td':
            if isinstance(tstring, str):
                # If one time window string is passed
                # default forecast naming
                fc_path = self.tree('forecasts', tstring)
                # default forecasts folder
                # A region must be given to the forecast
                return csep.load_catalog_forecast(fc_path, region=region)

    def create_forecast(self, tstring: str,
                        **kwargs) -> None:
        """

        Creates a forecast from the model source and a given time window

        Note:
            The argument `tstring` is formatted according to how the Experiment
            handles timewindows, specified in the functions
            :func:'floatcsep.utils.timewindow2str` and
            :func:'floatcsep.utils.str2timewindow`

        Args:
            tstring: String representing the start and end of the forecast,
                formatted as 'YY1-MM1-DD1_YY2-MM2-DD2'.
            **kwargs:

        """
        start_date, end_date = str2timewindow(tstring)
        # Model src is a file
        if self.model_class == 'ti':
            self.forecast_from_file(start_date, end_date, **kwargs)
        # Model src is a func or binary
        else:
            fc_path = self.tree('forecasts', tstring)
            if kwargs.get('force') or not os.path.exists(fc_path):
                self.forecast_from_func(start_date, end_date,
                                        **self.func_kwargs,
                                        **kwargs)
            else:
                print('Forecast already exists')

    def forecast_from_file(self, start_date: datetime, end_date: datetime,
                           **kwargs) -> None:
        """

        Generates a forecast from a file, by parsing and scaling it to
        the desired time window. H

        Args:
            start_date (~datetime.datetime): Start of the forecast
            end_date (~datetime.datetime): End of the forecast
            **kwargs: Keyword arguments for
             :class:`csep.core.forecasts.GriddedForecast`

        """

        time_horizon = decimal_year(end_date) - decimal_year(start_date)
        tstring = timewindow2str([start_date, end_date])

        f_parser = getattr(ForecastParsers, self.fmt)
        rates, region, mags = f_parser(self.path)

        forecast = GriddedForecast(
            name=f'{self.name}',
            data=rates,
            region=region,
            magnitudes=mags,
            start_time=start_date,
            end_time=end_date
        )

        scale = time_horizon / self.forecast_unit
        if scale != 1.0:
            forecast = forecast.scale(scale)

        print(
            f"Forecast expected count: {forecast.event_count:.2f}"
            f" with scaling parameter: {time_horizon:.1f}")

        self.forecasts[tstring] = forecast

    def forecast_from_func(self, start_date: datetime, end_date: datetime,
                           **kwargs) -> None:

        self.prepare_args(start_date, end_date, **kwargs)

        self.run_model()

    def prepare_args(self, start, end, **kwargs):

        filepath = os.path.join(self.path, self.tree('args'))
        fmt = os.path.splitext(filepath)[1]

        if fmt == '.txt':
            def replace_arg(arg, val, fp):
                with open(fp, 'r') as file_:
                    lines = file_.readlines()

                pattern_exists = False
                for k, line in enumerate(lines):
                    if line.startswith(arg):
                        lines[k] = f"{arg} = {val}\n"
                        pattern_exists = True
                        break  # Assuming there's only one occurrence of the key
                if not pattern_exists:
                    lines.append(f"{arg} = {val}\n")
                with open(fp, 'w') as file:
                    file.writelines(lines)

            replace_arg('start_date', start.isoformat(), filepath)
            replace_arg('end_date', end.isoformat(), filepath)
            for i, j in kwargs.items():
                replace_arg(i, j, filepath)
        elif fmt == '.json':
            with open(filepath, 'r') as file_:
                args = json.load(file_)
            args['start_date'] = start.isoformat()
            args['end_date'] = end.isoformat()

            args.update(kwargs)

            with open(filepath, 'w') as file_:
                json.dump(args, file_, indent=2)


    def run_model(self):

        if self.build == 'pip' or self.build == 'venv':
            print(f'Running model {self.name} using venv')
            run_func = f'{self.func} {self.tree("args")}'
            cmd = ['bash', '-c',
                   f'{self.run_prefix} {run_func}']
            process = subprocess.Popen(cmd,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       universal_newlines=True)
            for line in process.stdout:
                print(f'\t{line}', end='')
            process.wait()

    def to_dict(self, excluded=('name', 'forecasts')):
        """

        Returns:
            Dictionary with relevant attributes. Model can be reinstantiated
            from this dict

        """
        def _get_value(x):
            # For each element type, transforms to desired string/output
            if hasattr(x, 'to_dict'):
                # e.g. csep region, model, test, etc.
                o = x.to_dict()
            else:
                try:
                    try:
                        o = getattr(x, '__name__')
                    except AttributeError:
                        o = getattr(x, 'name')
                except AttributeError:
                    if isinstance(x, numpy.ndarray):
                        o = x.tolist()
                    else:
                        o = x
            return o

        def iter_attr(val):
            # recursive iter through nested dicts/lists
            if isinstance(val, Mapping):
                return {item: iter_attr(val_) for item, val_ in val.items()
                        if ((item not in excluded) and val_)}
            elif isinstance(val, Sequence) and not isinstance(val, str):
                return [iter_attr(i) for i in val]
            else:
                return _get_value(val)

        listwalk = [(i, j) for i, j in self.__dict__.items() if
                    not i.startswith('_')]

        dictwalk = {i: j for i, j in listwalk}
        # if self.model_config is None:
        #     dictwalk['models'] = iter_attr(self.models)
        # if self.test_config is None:
        #     dictwalk['tests'] = iter_attr(self.tests)

        return {self.name: iter_attr(dictwalk)}

    @classmethod
    def from_dict(cls, record: dict, **kwargs):
        """
        Returns a Model instance from a dictionary containing the required
        atrributes. Can be used to quickly instantiate from a .yml file.


        Args:
            record (dict): Contains the keywords from the ``__init__`` method.

                Note:
                    Must have either an explicit key `name`, or it must have
                    exactly one key with the model's name, whose values are
                    the remaining ``__init__`` keywords.

        Returns:
            A Model instance
        """

        if 'name' in record.keys():
            return cls(**record)
        elif len(record) != 1:
            raise IndexError('A single model has not been passed')
        name = next(iter(record))
        return cls(name=name, **record[name], **kwargs)
