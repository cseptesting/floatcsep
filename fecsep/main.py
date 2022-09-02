import sys
import datetime
import os
from fecsep import models
from collections import defaultdict


def run(cfg, test_date_string):

    exp = models.Experiment.from_yaml(cfg)
    exp.set_tests()
    exp.set_models()
    exp.stage_models()
    try:
        test_date = datetime.datetime.strptime(test_date_string, '%Y-%m-%dT%H:%M:%S')
    except:
        raise RuntimeError("Error parsing test date string. Should have format='%Y-%m-%dT%H:%M:%S")
    exp.set_test_date(test_date)

    print('Experiment is configured with the following parameters')
    print('======================================================\n')
    print(exp.to_yaml())

    # 0. create expected paths based on configuration
    exp.get_run_struct()

    # 1. retrieve data from isc catalog reader or from previous run
    catalog = exp.get_catalog()
    if not os.path.exists(exp.target_paths['catalog']):
        catalog.write_json(exp.target_paths['catalog'])

    # 2. Prepare each test that needs to be computed, could be oom problem for large sized tests
    test_list = exp.prepare_all_tests()

    # 3. Evaluate models
    run_results = defaultdict(list)
    for test in test_list:
        result = exp.run_test(test)
        run_results[test.name].append(result)

    # 4. Plot results
    exp.plot_results(run_results)

    # 5. Generate report
    exp.generate_report()

def fecsep():
    """

    """
    args = sys.argv[1:]
    try:
        func = globals()[args[0]]
    except AttributeError:
        raise AttributeError('Function not implemented')

    func(*args[1:])

