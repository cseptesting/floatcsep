
from floatcsep.experiment import Experiment

file_ = 'config.yml'
exp_c = Experiment.from_yml(file_)
exp_c.set_models()
exp_c.set_tests()
exp_c.set_tasks()
exp_c.run()
exp_c.plot_results()
exp_c.plot_forecasts()
exp_c.generate_report()
