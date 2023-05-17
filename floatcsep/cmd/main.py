from floatcsep.experiment import Experiment
import argparse

def stage(config, show=True):

    exp = Experiment.from_yml(config)
    exp.stage_models()
    print('\n==================')
    print('Staging experiment')
    print('==================\n')


def run(config, show=True):
    exp = Experiment.from_yml(config)

    exp.stage_models()
    print('\n==================')
    print('Running experiment')
    print('==================\n')
    exp.set_tasks()
    exp.run()
    print('\n================')
    print('Calculation done')
    print('================\n')

    if show:
        print('\n=============================')
        print("Plotting experiment's results")
        print('=============================\n')
        exp.plot_results()
        exp.plot_forecasts()
        exp.generate_report()

    print('\n========')
    print('Complete')
    print('========\n')


def plot(config, **_):
    exp = Experiment.from_yml(config)
    exp.stage_models()
    exp.set_tasks()
    print('\n=============================')
    print("Plotting experiment's results")
    print('=============================\n')
    exp.plot_results()
    exp.plot_forecasts()
    exp.generate_report()


def reproduce(config, show=True):
    exp = Experiment.from_yml(config)

    exp.stage_models()
    print('\n==================')
    print('Re-running experiment')
    print('==================\n')
    exp.set_tasks(run_name='reproducibility')
    exp.run()
    print('\n================')
    print('Calculation done')
    print('================\n')

    if show:
        print('\n=============================')
        print("Recreating experiment's figures")
        print('=============================\n')
        exp.plot_results()
        exp.plot_forecasts()
        exp.generate_report()

    print('\n========')
    print('Complete')
    print('========\n')


def floatcsep():
    """

    """
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('func', type=str, choices=['run', 'stage',
                                                   'plot', 'reproduce'],
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
