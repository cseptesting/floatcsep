import importlib
import fecsep.experiment
import fecsep.utils
import fecsep.accessors
from fecsep.experiment import Experiment

file_ = 'config.yml'

a = Experiment.from_yml(file_)
a.set_models()
a.set_tests()

a.prepare_paths()
a.prepare_tasks()
a.run()
a.plot_results()
a.generate_report()
