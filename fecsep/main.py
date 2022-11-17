import sys
import datetime
from fecsep import core
from collections import defaultdict
import argparse
import os


def run(config, test_date=None, use_saved=False):
    exp = core.Experiment.from_yaml(config)
    exp.set_test_date(test_date)

    exp.set_tests()
    exp.set_models()
    exp.prepare_paths()
    print('Experiment is configured with the following parameters')
    print('======================================================\n')
    print(exp.target_paths)
    print(exp.exists)
    exp.prepare_tasks()
    exp.stage_models()

    exp.get_run_struct()
    #
    exp.get_catalog()

    test_list = exp.prepare_all_tests()
    if use_saved is False:
        run_results = defaultdict(list)
        for test in test_list:
            result = exp.run_test(test)
            run_results[test.name].append(result)
    else:  # todo: need more elegant way of self-discovery
        run_results = defaultdict(list)
        for test in test_list:
            run_results[test.name] = exp.read_evaluation_result(test, exp.models, exp.target_paths)

    exp.plot_results(run_results)

    exp.generate_report()


def fecsep():
    """

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('func', type=str, choices=['run'],
                        help='Run a calculation')
    parser.add_argument('config', type=str,
                        help='Experiment Configuration file')
    parser.add_argument('-t', '--test_date', type=str, help='Date to test',
                        default=None)
    parser.add_argument('-s', '--use_saved', type=str,
                        help='Use saved results', default=False)

    args = parser.parse_args()
    try:
        func = globals()[args.func]
        args.__delattr__('func')
    except AttributeError:
        raise AttributeError('Function not implemented')
    func(**vars(args))


if __name__ == '__main__':
    test_run = '../tests/data_tests/gefe_qtree/'
    os.chdir(test_run)
    run('config.yml')
