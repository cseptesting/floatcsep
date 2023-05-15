from datetime import datetime, timedelta
from fecsep.experiment import Experiment
from fecsep.utils import timewindow2str


config = 'config.yml'


exp = Experiment.from_yml(config)
exp.stage_models()

a = exp.models[0]

start = datetime(2020, 1, 2)
end = datetime(2020, 1, 3)
a.forecast_from_func(start, end, n_sims=10000)

twstr = timewindow2str([start, start + timedelta(1)])
b = a.get_forecast(twstr, exp.region)
b.plot()

from matplotlib import pyplot
pyplot.show()
print('\n==================')
print('Staging experiment')
print('==================\n')
