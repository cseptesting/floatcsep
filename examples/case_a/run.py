
from fecsep.experiment import Experiment

file_ = 'config.yml'

exp_a = Experiment.from_yml(file_)
# exp_a.set_models()
# exp_a.set_tests()
exp_a.set_tasks()
exp_a.run()
exp_a.plot_results()
exp_a.generate_report()
exp_a.to_yml('test', extended=True)
