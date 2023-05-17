
from floatcsep.experiment import Experiment
from csep.utils.plots import plot_number_test

config = 'config.yml'


exp = Experiment.from_yml(config)
exp.stage_models()

# m = exp.models[0]
exp.set_tasks()
exp.run()

cat = exp.catalog
dt = [i for i in cat.get_datetimes()]

from datetime import datetime
start = datetime.combine(exp.start_date.date(), exp.start_date.time(), tzinfo=dt[0].tzinfo)
end = datetime.combine(exp.end_date.date(), exp.end_date.time(), tzinfo=dt[0].tzinfo)

print(cat.get_number_of_events())
test_cat = cat.filter([f'origin_time >= {start.timestamp()*1000}',
                       f'origin_time < {end.timestamp()*1000}'],
                      in_place=False)
print(test_cat.get_number_of_events())
test_cat2 = test_cat.filter([f'magnitude >= 3.5'],
                      in_place=False)
print(test_cat2.get_number_of_events())
print('\n================')
print('Calculation done')
# print('\n=============================')
# print("Plotting experiment's results")
# print('=============================\n')
exp.plot_results()
# exp.plot_forecasts()
# exp.generate_report()
#
# res = exp.read_results(exp.tests[0], exp.timewindows[-1])
# plot_number_test(res[0])