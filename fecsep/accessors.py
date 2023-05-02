from datetime import datetime
from urllib import request as request_
from urllib.parse import urlencode
from csep.utils.time_utils import datetime_to_utc_epoch
from csep.core.catalogs import CSEPCatalog
import xml.etree.ElementTree as ElementTree
from git import Repo, InvalidGitRepositoryError, NoSuchPathError
import time
import requests
import hashlib
import os
import sys
import shutil

HOST_CATALOG = "http://www.isc.ac.uk/cgi-bin/web-db-run?"
TIMEOUT = 180


def query_isc_gcmt(start_time: datetime, end_time: datetime,
                   min_magnitude: float = 5.0,
                   min_depth: float = None, max_depth: float = None,
                   min_latitude: float = None, max_latitude: float = None,
                   min_longitude: float = None, max_longitude: float = None,
                   catalog_id: str = None,
                   verbose: bool = False) -> CSEPCatalog:
    """
    Queries the International Seismological Service (http://www.isc.ac.uk) API
    to retrieve the global CMT catalogue (https://www.globalcmt.org/) in
    QuakeML format, which is then converted to a CSEPCatalog

    Args:
        start_time (datetime.datetime): start date-time of the query
        end_time (datetime.datime): end date-time of the query
        min_magnitude (float): cutoff magnitude
        min_depth (float): minimum depth
        max_depth (float): maximum depth
        min_latitude (float): minimum latitude
        max_latitude (float): maximum latitude
        min_longitude (float): minimum longitude
        max_longitude (float): maximum longitude
        catalog_id (str): identifier assigned to the catalog
        verbose (bool): flag to print log
    Returns:
        CSEPCatalog
    """

    if min_latitude:
        searchshape = 'RECT'
    else:
        searchshape = 'GLOBAL'
    events, creation_time = _query_isc_gcmt(
        start_year=start_time.year,
        start_month=start_time.month,
        start_day=start_time.day,
        searchshape=searchshape,
        start_time=start_time.time().isoformat(),
        end_year=end_time.year,
        end_month=end_time.month,
        end_day=end_time.day,
        end_time=end_time.time().isoformat(),
        min_mag=min_magnitude,
        min_dep=min_depth,
        max_dep=max_depth,
        left_lon=min_longitude,
        right_lon=max_longitude,
        bot_lat=min_latitude,
        top_lat=max_latitude,
        verbose=verbose
    )
    catalog = CSEPCatalog(data=events, name='ISC Bulletin - gCMT',
                          catalog_id=catalog_id, date_accessed=creation_time)
    catalog.filter([f'magnitude >= {min_magnitude}'], in_place=True)
    return catalog


def _query_isc_gcmt(out_format: str = 'QuakeML',
                    request: str = 'COMPREHENSIVE',
                    searchshape: str = 'GLOBAL',
                    start_year: int = 2020,
                    start_month: int = 1,
                    start_day: int = 1,
                    start_time: str = '00:00:00',
                    end_year: int = 2022,
                    end_month: int = 1,
                    end_day: int = 1,
                    end_time: str = '23:59:59',
                    host: str = None,
                    include_magnitudes: str = 'on',
                    min_mag: float = 5.95,
                    max_mag: float = None,
                    min_dep: float = None,
                    max_dep: float = None,
                    left_lon: float = None,
                    right_lon: float = None,
                    bot_lat: float = None,
                    top_lat: float = None,
                    req_mag_type: str = 'MW',
                    req_mag_agcy: str = 'GCMT',
                    verbose: bool = False):
    """
    Formats the query url by the search parameters.

    Args:
        out_format (str): 'QuakeML' (recommended) or 'ISF'
        request (str): 'COMPREHENSIVE' or 'REVIEWED' by ISC analyst
        searchshape (str): 'GLOBAL' or 'RECT'. Other options not imp. in csep
        host (str): Host to do the call. Uses HOST_CATALOG defined atop module
        include_magnitudes (str): 'on' for csep purposes
        req_mag_type (str): 'MW' for GCMT
        req_mag_agcy (str): 'GCMT'
        verbose (bool): print log

    Returns:

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
        print(
            f'\tCatalog with {len(events)} events downloaded in '
            f'{(time.time() - start_time):.2f} seconds')

    return events, creation_time


def _search_isc_gcmt(**newargs):
    """
    Performs de query at ISC API and returns event list and access date

    """
    paramstr = urlencode(newargs)
    url = HOST_CATALOG + paramstr
    try:
        fh = request_.urlopen(url, timeout=TIMEOUT)
        data = fh.read().decode('utf8')
        fh.close()
        root = ElementTree.fromstring(data)
        ns = root[0].tag.split('}')[0] + '}'
        creation_time = root[0].find(ns + 'creationInfo').find(
            ns + 'creationTime').text
        creation_time = creation_time.replace('T', ' ')
        events_quakeml = root[0].findall(ns + 'event')
        events = []
        for feature in events_quakeml:
            events.append(_parse_isc_event(feature, ns))

    except ElementTree.ParseError as msg:
        raise Exception('Badly-formed URL. "%s"' % msg)

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
    mag_gcmt = [i for i in magnitudes if
                i.find(ns + 'creationInfo')[0].text == mag_author][0]

    origin_id = mag_gcmt.find(ns + 'originID').text
    origins = node.findall(ns + 'origin')

    origin_gcmt = [i for i in origins if i.attrib['publicID'] == origin_id][0]

    lat = origin_gcmt.find(ns + 'latitude').find(ns + 'value').text
    lon = origin_gcmt.find(ns + 'longitude').find(ns + 'value').text
    mag = mag_gcmt.find(ns + 'mag').find(ns + 'value').text
    depth = origin_gcmt.find(ns + 'depth').find(ns + 'value').text

    dtstr = origin_gcmt.find(ns + 'time').find(ns + 'value').text
    date = dtstr.split('T')[0]
    time_ = dtstr.split('T')[1][:-1]
    dtime = datetime_to_utc_epoch(
        datetime.fromisoformat(date + ' ' + time_ + '0'))

    return id_, dtime, float(lat), float(lon), float(depth) / 1000., float(mag)


def _download_file(url: str, filename: str) -> None:
    """

    Downloads files (from zenodo)

    Args:
        url (str):
        filename (str):

    """
    progress_bar_length = 72
    block_size = 1024

    r = requests.get(url, stream=True)
    total_size = r.headers.get('content-length', False)
    if not total_size:
        with requests.head(url) as h:
            try:
                total_size = int(h.headers.get('Content-Length', 0))
            except TypeError:
                total_size = 0
    else:
        total_size = int(total_size)
    download_size = 0
    if total_size:
        print(
            f'Downloading file with size of {total_size / block_size:.3f} kB')
    else:
        print(f'Downloading file with unknown size')
    with open(filename, 'wb') as f:
        for data in r.iter_content(chunk_size=block_size):
            download_size += len(data)
            f.write(data)
            if total_size:
                progress = int(
                    progress_bar_length * download_size / total_size)
                sys.stdout.write(
                    '\r[{}{}] {:.1f}%'.format('█' * progress, '.' *
                                              (progress_bar_length - progress),
                                              100 * download_size / total_size)
                )
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


def from_zenodo(record_id, folder, force=False):
    """
    Download data from a Zenodo repository.
    Downloads if file does not exist, checksum has changed in local respect to
    url or force

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
                print(
                    f"Checksum is different: re-downloading {fname}"
                    f" from Zenodo...")
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


def from_git(url, path, branch=None, depth=1, **kwargs):
    """

    Clones a shallow repository from a git url

    Args:
        url (str): url of the repository
        path (str): path/folder where to clone the repo
        branch (str): repository's branch to clone (default: main)
        depth (int): depth history of commits
        **kwargs: keyword args passed to Repo.clone_from

    Returns:
        the pygit repository
    """

    kwargs.update({'depth': depth})
    try:
        repo = Repo(path)
    except (NoSuchPathError, InvalidGitRepositoryError):
        repo = Repo.clone_from(url, path, branch=branch, **kwargs)
        git_dir = os.path.join(path, '.git')
        if os.path.isdir(git_dir):
            shutil.rmtree(git_dir)

    return repo
