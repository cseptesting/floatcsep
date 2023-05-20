from datetime import datetime
from urllib import request
from urllib.parse import urlencode
from csep.utils.time_utils import datetime_to_utc_epoch, utc_now_datetime
from csep.core.catalogs import CSEPCatalog
import git
import requests
import hashlib
import os
import sys
import shutil

HOST_CATALOG = "https://service.iris.edu/fdsnws/event/1/query?"
TIMEOUT = 180


def query_gcmt(start_time, end_time, min_magnitude=5.0,
               max_depth=None,
               catalog_id=None,
               min_latitude=None, max_latitude=None,
               min_longitude=None, max_longitude=None):

    eventlist = _query_gcmt(start_time=start_time,
                            end_time=end_time,
                            min_magnitude=min_magnitude,
                            min_latitude=min_latitude,
                            max_latitude=max_latitude,
                            min_longitude=min_longitude,
                            max_longitude=max_longitude,
                            max_depth=max_depth)

    catalog = CSEPCatalog(data=eventlist,
                          name='gCMT',
                          catalog_id=catalog_id,
                          date_accessed=utc_now_datetime())
    return catalog


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
    git.refresh()

    try:
        repo = git.Repo(path)
    except (git.NoSuchPathError, git.InvalidGitRepositoryError):
        repo = git.Repo.clone_from(url, path, branch=branch, **kwargs)
        git_dir = os.path.join(path, '.git')
        if os.path.isdir(git_dir):
            shutil.rmtree(git_dir)

    return repo


def _query_gcmt(start_time, end_time, min_magnitude=3.50,
                min_latitude=None, max_latitude=None,
                min_longitude=None, max_longitude=None,
                max_depth=1000, extra_gcmt_params=None):
    """
    Return GCMT eventlist from IRIS web service.
    For details see "https://service.iris.edu/fdsnws/event/1/"
    Args:
        start_time (datetime.datetime): start time of catalog query
        end_time (datetime.datetime): end time of catalog query
        min_magnitude (float): minimum magnitude of query
        min_latitude (float): minimum latitude of query
        max_latitude (float): maximum latitude of query
        min_longitude (float): minimum longitude of query
        max_longitude (float): maximum longitude of query
        max_depth (float): maximum depth of query
        extra_gcmt_params (dict): additional parameters to pass to IRIS search
         function

    Returns:
        eventlist
    """
    extra_gcmt_params = extra_gcmt_params or {}

    eventlist = gcmt_search(minmagnitude=min_magnitude,
                            minlatitude=min_latitude,
                            maxlatitude=max_latitude,
                            minlongitude=min_longitude,
                            maxlongitude=max_longitude,
                            starttime=start_time.isoformat(),
                            endtime=end_time.isoformat(),
                            maxdepth=max_depth, **extra_gcmt_params)

    return eventlist

def gcmt_search(format='text',
                starttime=None,
                endtime=None,
                updatedafter=None,
                minlatitude=None,
                maxlatitude=None,
                minlongitude=None,
                maxlongitude=None,
                latitude=None,
                longitude=None,
                maxradius=None,
                catalog='GCMT',
                contributor=None,
                maxdepth=1000,
                maxmagnitude=10.0,
                mindepth=-100,
                minmagnitude=0,
                offset=1,
                orderby='time-asc',
                host=None,
                verbose=False):
    """Search the IRIS database for events matching input criteria.
    This search function is a wrapper around the ComCat Web API described here:
    https://service.iris.edu/fdsnws/event/1/

    This function returns a list of SummaryEvent objects, described elsewhere in this package.
    Args:
        starttime (datetime):
            Python datetime - Limit to events on or after the specified start time.
        endtime (datetime):
            Python datetime - Limit to events on or before the specified end time.
        updatedafter (datetime):
           Python datetime - Limit to events updated after the specified time.
        minlatitude (float):
            Limit to events with a latitude larger than the specified minimum.
        maxlatitude (float):
            Limit to events with a latitude smaller than the specified maximum.
        minlongitude (float):
            Limit to events with a longitude larger than the specified minimum.
        maxlongitude (float):
            Limit to events with a longitude smaller than the specified maximum.
        latitude (float):
            Specify the latitude to be used for a radius search.
        longitude (float):
            Specify the longitude to be used for a radius search.
        maxradius (float):
            Limit to events within the specified maximum number of degrees
            from the geographic point defined by the latitude and longitude parameters.
        catalog (str):
            Limit to events from a specified catalog.
        contributor (str):
            Limit to events contributed by a specified contributor.
        maxdepth (float):
            Limit to events with depth less than the specified maximum.
        maxmagnitude (float):
            Limit to events with a magnitude smaller than the specified maximum.
        mindepth (float):
            Limit to events with depth more than the specified minimum.
        minmagnitude (float):
            Limit to events with a magnitude larger than the specified minimum.
        offset (int):
            Return results starting at the event count specified, starting at 1.
        orderby (str):
            Order the results. The allowed values are:
            - time order by origin descending time
            - time-asc order by origin ascending time
            - magnitude order by descending magnitude
            - magnitude-asc order by ascending magnitude
        host (str):
            Replace default ComCat host (earthquake.usgs.gov) with a custom host.
    Returns:
        list: List of SummaryEvent() objects.
    """

    # getting the inputargs must be the first line of the method!
    inputargs = locals().copy()
    newargs = {}

    for key, value in inputargs.items():
        if value is True:
            newargs[key] = 'true'
            continue
        if value is False:
            newargs[key] = 'false'
            continue
        if value is None:
            continue
        newargs[key] = value

    del newargs['verbose']

    events = _search_gcmt(**newargs)

    return events


def _search_gcmt(**_newargs):
    """
    Performs de-query at ISC API and returns event list and access date

    """
    paramstr = urlencode(_newargs)
    url = HOST_CATALOG + paramstr
    fh = request.urlopen(url, timeout=TIMEOUT)
    data = fh.read().decode('utf8').split('\n')
    fh.close()
    eventlist = []
    for line in data[1:]:
        line_ = line.split('|')
        if len(line_) != 1:
            id_ = line_[0]
            time_ = datetime.fromisoformat(line_[1])
            dt = datetime_to_utc_epoch(time_)
            lat = float(line_[2])
            lon = float(line_[3])
            depth = float(line_[4])
            mag = float(line_[10])
            eventlist.append((id_, dt, lat, lon, depth, mag))

    return eventlist


def _download_file(url: str, filename: str) -> None:
    """

    Downloads files (from zenodo)

    Args:
        url (str): the url where the file is located
        filename (str): the filename required.

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
