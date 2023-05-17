
from floatcsep.experiment import Experiment


config = 'config.yml'


exp = Experiment.from_yml(config)
exp.stage_models()

# m = exp.models[0]
exp.set_tasks()
exp.run()
# print('\n================')
# print('Calculation done')
# # print('\n=============================')
# # print("Plotting experiment's results")
# # print('=============================\n')
# exp.plot_results()
# exp.plot_forecasts()
# exp.generate_report()