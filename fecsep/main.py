import sys
import datetime
from fecsep import core
from collections import defaultdict


def run(cfg, test_date_string=None, rerun=False):


    exp = core.Experiment.from_yaml(cfg)
    if test_date_string:
        try:
            test_date = datetime.datetime.strptime(test_date_string, '%Y-%m-%dT%H:%M:%S')
        except:
            raise RuntimeError("Error parsing test date string. Should have format='%Y-%m-%dT%H:%M:%S")
        exp.set_test_date(test_date)
    else:
        exp.set_test_date(exp.end_date)

    print('Experiment is configured with the following parameters')
    print('======================================================\n')
    print(exp.to_yaml())

    exp.set_tests()

    exp.set_models()

    exp.stage_models()

    exp.get_run_struct()

    exp.get_catalog()

    test_list = exp.prepare_all_tests()
    if rerun is False:
        run_results = defaultdict(list)
        for test in test_list:
            result = exp.run_test(test)
            run_results[test.name].append(result)
    else:                                            #todo: need more elegant way of self-discovery
        run_results = defaultdict(list)
        for test in test_list:
            run_results[test.name] = exp.read_evaluation_result(test, exp.models, exp.target_paths)

    exp.plot_results(run_results)

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

