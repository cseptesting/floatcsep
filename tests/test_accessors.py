import os.path
import vcr
from datetime import datetime
from fecsep.accessors import query_isc_gcmt, _query_isc_gcmt, from_zenodo, \
    from_git, _check_hash
import unittest
from unittest import mock

root_dir = os.path.dirname(os.path.abspath(__file__))


def isc_gcmt_dir():
    data_dir = os.path.join(root_dir, 'artifacts', 'isc_gcmt')
    return data_dir


def zenodo_dir():
    data_dir = os.path.join(root_dir, 'artifacts', 'zenodo')
    return data_dir


def test_isc_gcmt_search():
    tape_file = os.path.join(isc_gcmt_dir(), 'vcr_search.yaml')
    with vcr.use_cassette(tape_file):
        # Maule, Chile
        eventlist = \
            _query_isc_gcmt(start_year=2010, start_month=2, start_day=26,
                            end_year=2010, end_month=2, end_day=28,
                            min_mag=8.5)[0]
        event = eventlist[0]
        assert event[0] == '14340585'


def test_isc_gcmt_summary():
    tape_file = os.path.join(isc_gcmt_dir(), 'vcr_summary.yaml')
    with vcr.use_cassette(tape_file):
        eventlist = \
            _query_isc_gcmt(start_year=2010, start_month=2, start_day=26,
                            end_year=2010, end_month=2, end_day=28,
                            min_mag=8.5)[0]
        event = eventlist[0]
        cmp = "('14340585', 1267252513600, -35.98, -73.15, 23.2, 8.78)"
        assert str(event) == cmp
        assert event[0] == '14340585'
        assert datetime.fromtimestamp(
            event[1] / 1000.) == datetime.fromtimestamp(1267252513.600)
        assert event[2] == -35.98
        assert event[3] == -73.15
        assert event[4] == 23.2
        assert event[5] == 8.78


def test_zenodo_query():
    from_zenodo(4739912, zenodo_dir())
    txt_file = os.path.join(zenodo_dir(), 'dummy.txt')
    tar_file = os.path.join(zenodo_dir(), 'dummy.txt')
    assert os.path.isfile(txt_file)
    assert os.path.isfile(tar_file)


def test_catalog_query_plot():
    start_datetime = datetime(2020, 1, 1)
    end_datetime = datetime(2021, 1, 1)
    fname = os.path.join(isc_gcmt_dir(), 'test_cat')
    catalog = query_isc_gcmt(start_time=start_datetime, end_time=end_datetime,
                             min_magnitude=5.95)
    catalog.plot(set_global=True, plot_args={'filename': fname})


def test_zenodo_files():
    txt_path = os.path.join(zenodo_dir(), 'dummy.txt')
    with open(os.path.join(txt_path), 'r') as dummy:
        assert dummy.readline() == 'test'
    tar_path = os.path.join(zenodo_dir(), 'dummy.tar')
    _check_hash(tar_path, 'md5:17f80d606ff085751998ac4050cc614c')


class TestGitter(unittest.TestCase):
    @mock.patch('fecsep.accessors.Repo')
    @mock.patch('fecsep.accessors.Git')
    def runTest(self, mock_git, mock_repo):
        p = mock.PropertyMock(return_value=False)
        type(mock_repo.clone_from.return_value).bare = p
        from_git(
            '/tmp/testrepo',
            'git@github.com:github/testrepo.git',
            'master'
        )
        mock_git.checkout.called_once_with('master')
