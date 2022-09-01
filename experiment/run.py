from fecsep import models
import os
from datetime import datetime as dt
from collections import defaultdict

exp_config = '../experiment/config.yml'
tests_config = '../experiment/tests.yml'
models_config = '../experiment/models.yml'
exp = models.Experiment.from_yaml(exp_config)
exp.set_tests()
exp.set_models()
exp.stage_models()

date = dt(2015, 1, 1, 0, 0, 0)
exp.set_test_date(date)

exp.get_run_struct()
catalog = exp.get_catalog()
if not os.path.exists(exp.target_paths['catalog']):
    catalog.write_json(exp.target_paths['catalog'])
test_list = exp.prepare_all_tests()

# 3. Evaluate models
run_results = defaultdict(list)
for test in test_list:
    result = exp.run_test(test)
    run_results[test.name].append(result)

# 4. Plot results
exp.plot_results(run_results)

# 4.1 Generate report
exp.generate_report()