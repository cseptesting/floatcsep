
from floatcsep.experiment import Experiment

file_ = 'config.yml'

exp_e = Experiment.from_yml(file_)
exp_e.set_tasks()
exp_e.run()
exp_e.plot_results()
exp_e.plot_forecasts()
exp_e.generate_report()
