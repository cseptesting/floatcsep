import os
import sys
import numpy
from pymock import libs
from copy import deepcopy


def main(arg_path=None, folder=None, verbose=False):
    """
    Main pymock's function
    Contains the main steps of creating a forecast for a given time window.

    1. Parse an argument file
    2. Reads the input catalog
    3. Creates the forecast as synthetic catalogs
    4. Writes the synthetic catalogs

    params:
        arg_path (str): Path to the input arguments file.
        folder (str): (Optional) Path to save output. Defaults to 'forecasts'
        verbose (bool): print log
    """

    # Create a forecasts folder in current directory if it does not exist.
    folder = folder or os.path.join(os.path.dirname(arg_path), '../forecasts')
    os.makedirs(folder, exist_ok=True)

    # 1. Gets input data and arguments.
    args = libs.read_args(arg_path)  # A dictionary containing parameters

    cat_path = args.get('catalog')
    n_sims = args.get('n_sims', 1000)  # Gets from args or default to 1000
    seed = args.get('seed', None)        # Gets from args or default to seed

    # 2. Reads input catalog
    catalog = libs.load_cat(path=cat_path)

    # 3. Run model
    forecast = make_forecast(catalog,
                             args,
                             n_sims=n_sims,
                             seed=seed,
                             verbose=verbose)

    # 4. Write forecasts
    libs.write_forecast(args['start_date'], args['end_date'], forecast, folder)


def make_forecast(input_catalog, args, n_sims=1000, seed=None, verbose=True):
    """
    Routine to create a forecast from an input catalog and argument dictionary

    Args:
        input_catalog (list): A CSEP formatted events list (see libs.load_cat)
        args (dict): Contains the arguments and its values
        n_sims (int): Number of stochastic catalogs to create
        seed (int): seed for random number generation
        verbose (bool): Flag to print out the logging.
    """
    start_date = args['start_date']
    end_date = args['end_date']
    dt = end_date - start_date
    mag_min = args.get('mag_min', 4.0)

    # set seed for pseudo-random number gen
    if seed:
        numpy.random.seed(seed)
    # filter catalog

    cat_total = [i for i in input_catalog if i[3] < start_date]
    catalog_prev = [i for i in cat_total if start_date - dt <= i[3] and
                    i[2] >= mag_min]

    # Previous time-window rate
    lambd = len(catalog_prev)
    # Background rate
    mu_total = len(cat_total) * (end_date - start_date) / (
            max([i[3] for i in cat_total]) - min([i[3] for i in cat_total]))

    # scale by GR with b=1
    obsmag_min = min([i[2] for i in cat_total])
    mu = mu_total * 10 ** (obsmag_min - mag_min)

    if verbose:
        print(
            f"Making forecast with model parameters:\n {args.__str__()}\n"
            f"and simulation parameters:\n"
            f" n_sims:{locals()['n_sims']}\n"
            f" seed:{locals()['seed']}")
        print(f'\tmu: {mu:.2e}\n\tlambda:{lambd:.2e}')

    # The model creates a random selection of N events from the input_catalog
    # A simulated catalog has N_events ~ Poisson(rate_prevday)
    forecast = []
    for n_cat in range(n_sims):
        n_events_bg = numpy.random.poisson(mu)
        idx_bg = numpy.random.choice(range(len(cat_total)), size=n_events_bg)

        n_events = numpy.random.poisson(lambd)
        idx = numpy.random.choice(range(len(catalog_prev)), size=n_events)

        random_cat = deepcopy([cat_total[i] for i in idx_bg])
        random_cat.extend(deepcopy([catalog_prev[i] for i in idx]))

        for i, event in enumerate(random_cat):
            # Positions remains the same as the random catalog
            # Get the magnitude value using GR with b=1
            mag_bins = numpy.arange(mag_min, 8.1, 0.1)
            prob_mag = 10 ** (-mag_bins[:-1]) - 10 ** (-mag_bins[1:])
            mag = numpy.random.choice(mag_bins[:-1],
                                      p=prob_mag / numpy.sum(prob_mag))
            event[2] = mag
            # For each event, assigns a random datetime between start and end:
            dt = numpy.random.random() * (
                    args['end_date'] - args['start_date'])
            event[3] = args['start_date'] + dt
            # Replace events and catalog ids
            event[5] = n_cat
            event[6] = i
            forecast.append(event)

    # if verbose:
    print(
        f'\tTotal of {len(forecast)} events M>{mag_min} in {n_sims}'
        f' synthetic catalogs')
    return forecast


def run():
    """
    Advanced usage for command entry point (see setup.cfg, entry_points)
    """
    args = sys.argv
    main(*args[1:])
