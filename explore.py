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

print('START1')
start = datetime(2020, 1, 1)
end = datetime(2021, 1, 1)
model.create_forecast(start, end)

with h5py.File('examples/case_f/model.hdf5', 'r') as a_:
    print(a_.keys())

print('START2')
start = datetime(2021, 1, 1)
end = datetime(2022, 1, 1)
model.create_forecast(start, end)

with h5py.File('examples/case_a/model.hdf5', 'r') as a_:
    key = a_.keys()
    print(key)
#
print('START3')
start = datetime(2021, 1, 1)
end = datetime(2023, 1, 1)
model.create_forecast(start, end)

with h5py.File('examples/case_a/model.hdf5', 'r') as a_:
    print(a_.keys())

model.rm_db()
