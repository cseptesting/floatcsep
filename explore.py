# from fecsep import experiment, registry, model, evaluation
from fecsep import experiment
import h5py
# reload(experiment)
# reload(registry)
# reload(model)
# reload(evaluation)
from fecsep.experiment import Experiment
from fecsep.model import Model
from datetime import datetime
import numpy

exp = Experiment.from_yml('examples/case_f/config.yml')

exp.set_models()
exp.stage_models()
# print(exp.models[0].reg)
# print(exp.reg.tree['Model C'])
model = exp.models[0]

# with h5py.File(model.path) as f_:
#     print(f_.keys())
# print('START1')
start = datetime(2020, 1, 1)
end = datetime(2021, 1, 1)
model.create_forecast(start, end)

# aa = model.to_dict()
# b = Model.from_dict(aa)
# print(b.path)
