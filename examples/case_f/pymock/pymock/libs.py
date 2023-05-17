from datetime import datetime, time
import os


def syncat_path(start, end, folder):
    """
    Returns the file path of a forecast based on its start and end dates.
    """
    midnight = time(0, 0, 0)
    if start.time() == midnight and end.time() == midnight:
        start = start.date()
        end = end.date()

    return os.path.join(folder,
                        f"pymock_{start.isoformat()}_{end.isoformat()}.csv"
                        )


def load_cat(path):
    """
    Loads a catalog forecast using the CSEP format

    Args:
        path (str): Path to the catalog file

    Returns:
        A list of CSEP formatted events in
            lon, lat, mag, time_str, depth, catalog_id, event_id
    """
    catalog = []
    with open(path) as f_:
        for line in f_.readlines()[1:]:
            line = line.split(',')
            event = [float(line[0]), float(line[1]), float(line[2]),
                     datetime.fromisoformat(line[3]),
                     float(line[4]), int(line[5]), int(line[6])]
            catalog.append(event)

    return catalog


def write_forecast(start, end, forecast, folder=None):
    """
    Writes a catalog forecast using the CSEP format  in
        lon, lat, mag, time_str, depth, catalog_id, event_id
    """

    if folder is None:
        folder = 'forecasts'
    os.makedirs(folder, exist_ok=True)
    with open(syncat_path(start, end, folder), 'w') as file_:
        file_.write('lon, lat, M, time_string, depth, catalog_id, event_id\n')
        for event in forecast:
            line = f'{event[0]},{event[1]},{event[2]:.2f},' \
                   f'{event[3].isoformat()},{event[4]},{event[5]},{event[6]}\n'
            file_.write(line)


def read_args(path):
    """
    Parses an arguments file. This file should be build as:
    {argument} = {argument_value}
    """
    import os, sys
    params = {'start_date': None, 'end_date': None}

    with open(path, 'r') as f_:
        for line in f_.readlines():
            line_ = [i.strip() for i in line.split('=')]
            if line_[0] == 'start_date':
                params['start_date'] = datetime.fromisoformat(line_[1])
            elif line_[0] == 'end_date':
                params['end_date'] = datetime.fromisoformat(line_[1])
            elif line_[0] == 'catalog':
                params['catalog'] = os.path.join(os.path.dirname(path),
                                                 line_[1])
            elif line_[0] == 'mag_min':
                params['mag_min'] = float(line_[1])
            elif line_[0] == 'n_sims':
                params['n_sims'] = int(line_[1])
            elif line_[0] == 'seed':
                params['seed'] = int(line_[1])

    return params
