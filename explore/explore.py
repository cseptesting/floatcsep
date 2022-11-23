import importlib
import fecsep.core
import fecsep.utils
import fecsep.accessors

importlib.reload(fecsep.core)
importlib.reload(fecsep.utils)
importlib.reload(fecsep.accessors)
from fecsep.core import Experiment

file_ = 'config.yml'

a = Experiment.from_yml(file_)
a.set_models()
a.set_tests()

# for i in a.tests:
#     print(i)
a.prepare_paths()
a.prepare_tasks()
a.run()
a.plot_results(show=True)
# a.plot_forecasts()
a.generate_report()
print('ready')
# for i in a.tasks:
#     i.run()
# file_ = 'config_out2.yaml'
# a.to_yaml(file_)
