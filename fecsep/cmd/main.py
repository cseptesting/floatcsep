from fecsep.experiment import Experiment
import argparse
import os


def run(config, show=True):
    exp = Experiment.from_yml(config)

    exp.set_tests()
    exp.set_models()
    exp.stage_models()
    exp.set_paths()
    print('\n==================')
    print('Running experiment')
    print('==================\n')
    exp.set_tasks()
    exp.run()

    if show:
        exp.plot_results()
        exp.plot_forecasts()
        exp.generate_report()
    print('\n================')
    print('Calculation done')
    print('================\n')


def plot(config, **_):
    exp = Experiment.from_yml(config)

    exp.set_tests()
    exp.set_models()
    exp.set_paths()
    print("Plotting experiment's results")
    print('=============================\n')
    exp.plot_results()
    exp.generate_report()


def fecsep():
    """

    """
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('func', type=str, choices=['run', 'plot'],
                        help='Run a calculation')
    parser.add_argument('config', type=str,
                        help='Experiment Configuration file')
    parser.add_argument('-s', '--show', type=str,
                        help='Use saved results', default=True)

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
