from floatcsep import __version__
from floatcsep.experiment import Experiment
import logging
import argparse
log = logging.getLogger(__name__)


def stage(config, show=True):

    log.info(f'Running floatCSEP v{__version__}')
    exp = Experiment.from_yml(config)
    exp.stage_models()
    log.info('Finalized')


def run(config, show=True):
    log.info(f'Running floatCSEP v{__version__}')

    exp = Experiment.from_yml(config)
    exp.stage_models()
    exp.set_tasks()
    exp.run()
    if show:
        exp.plot_results()
        exp.plot_forecasts()
        exp.generate_report()
    log.info('Finalized')


def plot(config, **_):

    log.info(f'Running floatCSEP v{__version__}')
    exp = Experiment.from_yml(config)
    exp.stage_models()
    exp.set_tasks()
    exp.plot_results()
    exp.plot_forecasts()
    exp.generate_report()
    log.info('Finalized')


def reproduce(config, show=True):

    log.info(f'Running floatCSEP v{__version__}')

    exp = Experiment.from_yml(config)
    exp.stage_models()
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
