"""
.. tutorial-case_a

First Experiment
================


"""

###############################################################################
# Load required libraries
# -----------------------
#
from floatcsep.experiment import Experiment

file_ = 'config.yml'

exp_a = Experiment.from_yml(file_)
exp_a.stage_models()
exp_a.set_tasks()
exp_a.run()
exp_a.plot_results()
exp_a.plot_forecasts()
exp_a.generate_report()
exp_a.make_repr()
