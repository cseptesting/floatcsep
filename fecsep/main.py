import sys
import datetime
from fecsep import core
from collections import defaultdict
import argparse
import os


def run(config, show=True):
    exp = core.Experiment.from_yml(config)

    exp.set_tests()
    exp.set_models()
    exp.prepare_paths()
    print('Running experiment')
    print('==================\n')
    exp.prepare_tasks()
    exp.run()

    if show:
        exp.plot_results()
        exp.plot_forecasts()
        exp.generate_report()


def plot(config, use_saved=False):
    exp = core.Experiment.from_yml(config)

    exp.set_tests()
    exp.set_models()
    exp.prepare_paths()
    print('Running experiment')
    print('======================================================\n')
    exp.plot_results()
    exp.generate_report()


def fecsep():
    """

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('func', type=str, choices=['run', 'plot'],
                        help='Run a calculation')
    parser.add_argument('config', type=str,
                        help='Experiment Configuration file')
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
