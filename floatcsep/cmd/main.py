from floatcsep import __version__
from floatcsep.experiment import Experiment
from floatcsep.utils import ExperimentComparison
import logging
import argparse
log = logging.getLogger('floatLogger')


def stage(config, **_):

    log.info(f'floatCSEP v{__version__} | Stage')
    exp = Experiment.from_yml(config)
    exp.stage_models()

    log.info('Finalized')
    log.debug('')


def run(config, **kwargs):

    log.info(f'floatCSEP v{__version__} | Run')
    exp = Experiment.from_yml(config, **kwargs)
    exp.stage_models()
    exp.set_tasks()
    exp.run()
    exp.plot_results()
    exp.plot_forecasts()
    exp.generate_report()
    exp.make_repr()

    log.info('Finalized')
    log.debug('')


def plot(config, **kwargs):

    log.info(f'floatCSEP v{__version__} | Plot')

    exp = Experiment.from_yml(config, **kwargs)
    exp.stage_models()
    exp.set_tasks()
    exp.plot_results()
    exp.plot_forecasts()
    exp.generate_report()

    log.info('Finalized')
    log.debug('')


def reproduce(config, **kwargs):

    log.info(f'floatCSEP v{__version__} | Reproduce')

    reproduced_exp = Experiment.from_yml(config, reprdir='reproduced',
                                         **kwargs)
    reproduced_exp.stage_models()
    reproduced_exp.set_tasks()
    reproduced_exp.run()

    original_config = reproduced_exp.original_config
    original_exp = Experiment.from_yml(original_config,
                                       rundir=reproduced_exp.original_rundir)
    original_exp.stage_models()
    original_exp.set_tasks()

    comp = ExperimentComparison(original_exp, reproduced_exp)
    comp.compare_results()

    log.info('Finalized')
    log.debug('')


def floatcsep():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('func', type=str, choices=['run', 'stage',
                                                   'plot', 'reproduce'],
                        help='Run a calculation')
    parser.add_argument('config', type=str,
                        help='Experiment Configuration file')
    parser.add_argument('-l', '--log', action='store_false', default=True,
                        help="Don't log experiment")
    parser.add_argument('-t', '--timestamp', action='store_true',
                        default=False, help="Timestamp results")
    args = parser.parse_args()
    try:
        func = globals()[args.func]
        args.__delattr__('func')

    except AttributeError:
        raise AttributeError('Function not implemented')
    func(**vars(args))
