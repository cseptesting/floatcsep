from datetime import datetime
from urllib import request
from urllib.parse import urlencode
from csep.utils.time_utils import datetime_to_utc_epoch
from csep.core.catalogs import CSEPCatalog
import xml.etree.ElementTree as ET
import time
import requests
import hashlib
import argparse
import os
import sys

HOST_CATALOG = "https://www.isc.ac.uk/cgi-bin/web-db-run?"
TIMEOUT = 180


def query_isc_gcmt(start_datetime, end_datetime, min_mw, min_depth=None, max_depth=None, cat_id=None, max_mw=None, verbose=False):

    events, creation_time = _query_isc_gcmt(start_year=start_datetime.year,
                                            start_month=start_datetime.month,
                                            start_day=start_datetime.day,
                                            start_time=start_datetime.time().isoformat(),
                                            end_year=end_datetime.year,
                                            end_month=end_datetime.month,
                                            end_day=end_datetime.day,
                                            end_time=end_datetime.time().isoformat(),
                                            min_mag=min_mw,
                                            max_mag=max_mw,
                                            min_dep=min_depth,
                                            max_dep=max_depth,
                                            verbose=verbose)

    return CSEPCatalog(data=events, name='ISC Bulletin - gCMT', catalog_id=cat_id, date_accessed=creation_time)


def _query_isc_gcmt(out_format='QuakeML',
                    request='COMPREHENSIVE',
                    searchshape='GLOBAL',
                    start_year=2020,
                    start_month=1,
                    start_day=1,
                    start_time='00:00:00',
                    end_year=2022,
                    end_month=1,
                    end_day=1,
                    end_time='23:59:59',
                    host=None,
                    include_magnitudes='on',
                    min_mag=5.95,
                    max_mag=None,
                    min_dep=None,
                    max_dep=None,
                    req_mag_type='MW',
                    req_mag_agcy='GCMT',
                    verbose=False):

    """ Return gCMT catalog from ISC online web-portal

        Args: 
            (follow csep.query_comcat for guidance)

        Returns:
            out (csep.core.catalogs.AbstractBaseCatalog): gCMT catalog
    """

    inputargs = locals().copy()
    query_args = {}
    for key, value in inputargs.items():
        if value is True:
            query_args[key] = 'true'
            continue
        if value is False:
            query_args[key] = 'false'
            continue
        if value is None:
            continue
        query_args[key] = value

    del query_args['verbose']

    start_time = time.time()
    if verbose:
        print('Accessing ISC API')

    events, creation_time, url = _search_isc_gcmt(**query_args)

    if verbose:
         print(f'\tAccess URL: {url}')
         print(f'\tCatalog with {len(events)} events downloaded in {(time.time() - start_time):.2f} seconds')


    return events, creation_time


def _search_isc_gcmt(**newargs):

    """
    Performs de query at ISC API and returns event list and access date

    """
    paramstr = urlencode(newargs)
    url = HOST_CATALOG + paramstr
    try:
        fh = request.urlopen(url, timeout=TIMEOUT)
        data = fh.read().decode('utf8')
        fh.close()
        root = ET.fromstring(data)
        ns = root.tag.split('}quakeml')[0] + '}'
        creation_time = root[0].find(ns + 'creationInfo').find(ns + 'creationTime').text
        creation_time = creation_time.replace('T', ' ')
        events_quakeml = root[0].findall(ns + 'event')
        events = []
        for feature in events_quakeml:
            events.append(_parse_isc_event(feature, ns))

    except Exception as msg:
        raise Exception(
            'Error downloading data from url %s.  "%s".' % (url, msg))

    return events, creation_time, url


def _parse_isc_event(node, ns, mag_author='GCMT'):
    """
    Parse event list from quakeML returned from ISC
    """
    id_ = node.get('publicID').split('=')[-1]
    magnitudes = node.findall(ns + 'magnitude')
    mag_gcmt = [i for i in magnitudes if i.find(ns + 'creationInfo')[0].text == mag_author][0]

    origin_id = mag_gcmt.find(ns + 'originID').text
    origins = node.findall(ns + 'origin')

    origin_gcmt = [i for i in origins if i.attrib['publicID'] == origin_id][0]

    lat = origin_gcmt.find(ns + 'latitude').find(ns + 'value').text
    lon = origin_gcmt.find(ns + 'longitude').find(ns + 'value').text
    mag = mag_gcmt.find(ns + 'mag').find(ns + 'value').text
    depth = origin_gcmt.find(ns + 'depth').find(ns + 'value').text

    dtstr = origin_gcmt.find(ns + 'time').find(ns + 'value').text
    date = dtstr.split('T')[0]
    time = dtstr.split('T')[1][:-1]
    dtime = datetime_to_utc_epoch(datetime.fromisoformat(date + ' ' + time + '0'))

    return (id_, dtime, float(lat), float(lon), float(depth) / 1000., float(mag))


def _download_file(url, filename):
    progress_bar_length = 72
    block_size = 1024

    r = requests.get(url, stream=True)
    if r.headers.get('content-length', False):
        with requests.head(url) as h:
            try:
                total_size = int(h.headers.get('Content-Length', 0))
            except TypeError:
                total_size = None
    download_size = 0
    if total_size:
        print(f'Downloading file with size of {total_size / block_size:.3f} kB')
    else:
        print(f'Downloading file with unknown size')
    with open(filename, 'wb') as f:
        for data in r.iter_content(chunk_size=block_size):
            download_size += len(data)
            f.write(data)
            if total_size:
                progress = int(progress_bar_length*download_size/total_size)
                sys.stdout.write('\r[{}{}] {:.1f}%'.format('█'*progress, '.' * (progress_bar_length-progress),
                    100*download_size/total_size))
                sys.stdout.flush()
        sys.stdout.write('\n')


def _check_hash(filename, checksum):
    """
    Checks if existing file hash matches checksum from url
    """
    algorithm, value = checksum.split(':')
    if not os.path.exists(filename):
        return value, 'invalid'
    h = hashlib.new(algorithm)
    with open(filename, 'rb') as f:
        while True:
            data = f.read(4096)
            if not data:
                break
            h.update(data)
    digest = h.hexdigest()
    return value, digest


def download_from_zenodo(record_id, folder, force=False):
    """
    Download data from a Zenodo repository.
    Downloads if file does not exist, checksum has changed in local respect to url
    or force
    Args:
        record_id: corresponding to the Zenodo repository
        folder: where the repository files will be downloaded
        force: force download even if file exists and checksum passes

    Returns:

    """
    # Grab the urls and filenames and checksums
    r = requests.get(f"https://zenodo.org/api/records/{record_id}")
    download_urls = [f['links']['self'] for f in r.json()['files']]
    filenames = [(f['key'], f['checksum']) for f in r.json()['files']]

    # Download and verify checksums
    for (fname, checksum), url in zip(filenames, download_urls):
        full_path = os.path.join(folder, fname)
        if os.path.exists(full_path):
            value, digest = _check_hash(full_path, checksum)
            if value != digest:
                print(f"Checksum is different: re-downloading {fname} from Zenodo...")
                _download_file(url, full_path)
            elif force:
                print(f"Re-downloading {fname} from Zenodo...")
                _download_file(url, full_path)
            else:
                print(f'Found file {fname}. Checksum OK.')

        else:
            print(f"Downloading {fname} from Zenodo...")
            _download_file(url, full_path)
        value, digest = _check_hash(full_path, checksum)
        if value != digest:
            print("Error: Checksum does not match")
            sys.exit(-1)

