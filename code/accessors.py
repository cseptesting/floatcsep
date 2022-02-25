from datetime import datetime
from urllib import request
from urllib.parse import urlencode
from csep.utils.time_utils import datetime_to_utc_epoch
from csep.core.catalogs import CSEPCatalog
import xml.etree.ElementTree as ET
import time

HOST = "http://www.isc.ac.uk/cgi-bin/web-db-run?"
TIMEOUT = 180


def query_isc_gcmt(start_datetime, end_datetime, min_mw, cat_id, max_mw=None, verbose=False):

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
    url = HOST + paramstr
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


def download_from_zenodo(resource_id):
    """ Downloads file from Zenodo and returns checksum for each file

        Note: Returns here should be a list even if one object is found

        Args:
            resource_id (int): resource id from zenodo

        Returns:
            checksum (string): md5 checksum from model
    """
    pass


