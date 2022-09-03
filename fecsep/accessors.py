from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from urllib import request
from urllib.parse import urlencode
from urllib.request import urlopen
from csep.utils.time_utils import datetime_to_utc_epoch
from csep.core.catalogs import CSEPCatalog
import xml.etree.ElementTree as ET
from git import Repo, InvalidGitRepositoryError, NoSuchPathError
import time
import requests
import hashlib
import os
import sys
import dateparser
import xmltodict
import json
from math import modf
from time import sleep


HOST_CATALOG = "http://www.isc.ac.uk/cgi-bin/web-db-run?"
TIMEOUT = 180


def query_isc_gcmt(start_datetime, end_datetime, min_mw, min_depth=None, max_depth=None,
                   cat_id=None, max_mw=None, verbose=False,
                   min_latitude=None, max_latitude=None,
                   min_longitude=None, max_longitude=None,
                   **kwargs,):
    if min_latitude:
        searchshape='RECT'
    else:
        searchshape='GLOBAL'
    events, creation_time = _query_isc_gcmt(start_year=start_datetime.year,
                                            start_month=start_datetime.month,
                                            start_day=start_datetime.day,
                                            searchshape=searchshape,
                                            start_time=start_datetime.time().isoformat(),
                                            end_year=end_datetime.year,
                                            end_month=end_datetime.month,
                                            end_day=end_datetime.day,
                                            end_time=end_datetime.time().isoformat(),
                                            min_mag=min_mw,
                                            max_mag=max_mw,
                                            min_dep=min_depth,
                                            max_dep=max_depth,
                                            left_lon=min_longitude,
                                            right_lon=max_longitude,
                                            bot_lat=min_latitude,
                                            top_lat=max_latitude,
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
                    left_lon=None,
                    right_lon=None,
                    bot_lat=None,
                    top_lat=None,
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

    return id_, dtime, float(lat), float(lon), float(depth) / 1000., float(mag)



def query_bsi(start_datetime, end_datetime, *args, **kwargs):
    """
    Queries INGV Bulletino Sismico Italiano, revised version.
    :return: csep.core.Catalog object
    """
    # DEFINED IN THE ITALY CSEP EXPERIMENT
    depthm = 30
    mmin = 2.5
    xmin = 5.0
    xmax = 20.0
    ymin = 35.0
    ymax = 40.0
    ################
    lon, lat, year, month, day, magL, depth, hour, minute, sec, type_mag, millisecond, ev_id = \
        [], [], [], [], [], [], [], [], [], [], [], [], []
    events, time_ev = [], []
    time_in = dateparser.parse(start_datetime)
    time_fm = dateparser.parse(end_datetime)
    time_fin = time_fm
    months = (time_fm.year - time_in.year) * 12 + time_fm.month - time_in.month
    for k in range(0, months + 1):
        if months == 0 and k == 0:
            first_part = '%04d-%02d-%02d' % (time_in.year, time_in.month, time_in.day)
            second_part = '%04d-%02d-%02d' % (time_fm.year, time_fm.month, time_fm.day)
        else:
            time_im = time_in + relativedelta(months=k)
            time_fm = time_in + relativedelta(months=k + 1)
            first_part = '%04d-%02d-%02d' % (time_im.year, time_im.month, time_im.day)
            second_part = '%04d-%02d-%02d' % (time_fm.year, time_fm.month, time_fm.day)
        if k == months and months != 0:
            time_im = time_in + relativedelta(months=k)
            time_fm = time_fin
            first_part = '%04d-%02d-%02d' % (time_im.year, time_im.month, time_im.day)
            second_part = '%04d-%02d-%02d' % (time_fm.year, time_fm.month, time_fm.day)
        start_datetime = str(first_part) + 'T00:00:00'
        end_datetime = str(second_part) + 'T00:00:00'
        HOST = "webservices.rm.ingv.it"
        if start_datetime != end_datetime:
            url_template = "http://" + HOST + "/fdsnws/event/1/query?starttime=" + str(
                start_datetime) + "&endtime=" + str(
                end_datetime) + "&minlat=" + str(ymin) + "&maxlat=" + str(ymax) + "&minlon=" + str(
                xmin) + "&maxlon=" + str(
                xmax) + "&maxdepth=" + str(depthm * 1000) + "&minmag=" + str(mmin) + "&format=geojson"
            print(url_template)
            html = urlopen(url_template).read()
            data = json.loads(html)
            field_list = data['features']
            i = 0
            for field in field_list:
                type_magnitude = data['features'][i]['properties']['magType']
                mag = data['features'][i]['properties']['mag']
                if type_magnitude == "Mw" or type_magnitude == "Md" or type_magnitude == "mb" or type_magnitude == "M":
                    id = int(data['features'][i]['properties']['eventId'])
                    url = "http://webservices.ingv.it/fdsnws/event/1/query?eventId=" + str(
                        id) + "&includeallmagnitudes=true"
                    print(url)
                    try:
                        html = urlopen(url).read()
                    except:
                        pass
                    sleep(0.1)
                    doc = xmltodict.parse(html, force_list=('magnitude',))
                    root = doc["q:quakeml"]
                    eventparameters = root["eventParameters"]
                    flag = 0
                    for data_ev in eventparameters["event"]['magnitude']:
                        mag_database = ((data_ev['mag']['value']))
                        type_database = ((data_ev['type']))
                        author = ((data_ev['creationInfo']['author']))
                        if type_database == "ML" and author == "Sala Operativa INGV-OE (Catania)":
                            mag = mag_database
                            mag_type = "ML"
                            flag = 1
                        if type_database == "ML" and author == "Manually reviewed by Franco Mele":
                            mag = mag_database
                            mag_type = "ML"
                            flag = 1
                        if type_database == "ML" and author == "Bollettino Sismico Italiano INGV":
                            mag = mag_database
                            mag_type = "ML"
                            flag = 1
                    if flag == 0:
                        if mag_type == "mb":
                            mag_type = "Mw"
                        if mag_type == "Md":
                            mag = round(float(mag) * 1.612 - 1.633, 1)
                        if mag_type == "Mw":
                            print('sono arrivato qui', id)
                            mag = round(0.93 * float(mag) + 0.164, 1)
                        if mag_type == "M":
                            mag = float(mag)
                        mag_type = "ML"
                else:
                    mag_type = data['features'][i]['properties']['magType']
                    mag = data['features'][i]['properties']['mag']
                mytime = data['features'][i]['properties']['time']
                time = dateparser.parse(mytime)
                ref_time = datetime(1970, 1, 1, tzinfo=timezone.utc)
                ev_id.append(int(data['features'][i]['properties']['eventId']))
                second = float(time.second) + (float(time.microsecond) / 1000000)
                lon.append(float(data['features'][i]['geometry']['coordinates'][0]))
                lat.append(float(data['features'][i]['geometry']['coordinates'][1]))
                depth.append(float(data['features'][i]['geometry']['coordinates'][2]))
                magL.append(mag)
                type_mag.append(mag_type)
                fraz, inter = modf(float(second))
                millisecond = (int(round(fraz, 3) * 1000000))
                sec = int(inter)
                time_aev = datetime(int(time.year), int(time.month), int(time.day), int(time.hour),
                                    int(time.minute), sec, millisecond, tzinfo=timezone.utc)
                time_ep = (time_aev - ref_time).total_seconds()
                time_ev.append(time_ep)
                i = i + 1
        nrows = len(time_ev) - 1
        for i in range(nrows, 0, -1):
            event_tuple = (
                ev_id[i],
                time_ev[i],
                float(lat[i]),
                float(lon[i]),
                float(depth[i]),
                float(magL[i])
            )
            events.append(event_tuple)
    return events



def _download_file(url, filename):
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


def from_zenodo(record_id, folder, force=False):
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


def from_git(url, path, branch=None, force=False):

    try:
        repo = Repo(path)
    except (NoSuchPathError, InvalidGitRepositoryError):
        repo = Repo.clone_from(url, path, branch=branch)

    return repo


