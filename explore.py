# from fecsep import experiment, registry, model, evaluation
from fecsep import experiment

#
# reload(experiment)
# reload(registry)
# reload(model)
# reload(evaluation)
from fecsep.experiment import Experiment

exp = Experiment.from_yml('examples/case_d/config.yml')

exp.set_tests()
exp.set_models()

exp.stage_models()
# print(exp.reg)
# print(Registry.number)
