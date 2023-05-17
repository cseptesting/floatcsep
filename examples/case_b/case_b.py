
from floatcsep.experiment import Experiment

file_ = 'config.yml'

exp_b = Experiment.from_yml(file_)
exp_b.set_models()
exp_b.set_tests()
exp_b.set_tasks()
exp_b.run()
exp_b.plot_results()
exp_b.generate_report()
