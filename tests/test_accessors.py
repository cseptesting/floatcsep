import os.path
import vcr
from datetime import datetime
from fecsep.accessors import _query_isc_gcmt, from_zenodo, from_git, _check_hash
import unittest
import mock


def get_datadir():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'artifacts', 'ISC_GCMT')
    return data_dir


def test_isc_gcmt_search():
    datadir = get_datadir()
    tape_file = os.path.join(datadir, 'vcr_search.yaml')
    with vcr.use_cassette(tape_file):
        # Maule, Chile
        eventlist = _query_isc_gcmt(start_year=2010, start_month=2, start_day=26,
                                    end_year=2010, end_month=2, end_day=28,
                                    min_mag=8.5)[0]
        event = eventlist[0]
        print(str(event))
        assert event[0] == '14340585'


def test_isc_gcmt_summary():
    datadir: str = get_datadir()
    tape_file = os.path.join(datadir, 'vcr_summary.yaml')
    with vcr.use_cassette(tape_file):
        eventlist = _query_isc_gcmt(start_year=2010, start_month=2, start_day=26,
                                    end_year=2010, end_month=2, end_day=28,
                                    min_mag=8.5)[0]
        event = eventlist[0]
        cmp = "('14340585', 1267252513600, -35.98, -73.15, 23.2, 8.78)"
        assert str(event) == cmp
        assert event[0] == '14340585'
        assert datetime.fromtimestamp(event[1] / 1000.) == datetime.fromtimestamp(1267252513.600)
        assert event[2] == -35.98
        assert event[3] == -73.15
        assert event[4] == 23.2
        assert event[5] == 8.78


def test_zenodo_query():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    zen_path = os.path.join(root_dir, 'artifacts', 'Zenodo')
    from_zenodo(4739912, zen_path)
    txt_file = os.path.join(zen_path, 'dummy.txt')
    tar_file = os.path.join(zen_path, 'dummy.txt')
    assert os.path.isfile(txt_file)
    assert os.path.isfile(tar_file)


def test_zenodo_files():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(root_dir, 'artifacts', 'Zenodo', 'dummy.txt')
    with open(os.path.join(txt_path), 'r') as dummy:
        assert dummy.readline() == 'test'
    tar_path = os.path.join(root_dir, 'artifacts', 'Zenodo', 'dummy.tar')
    _check_hash(tar_path, 'md5:17f80d606ff085751998ac4050cc614c')


class test_gitter(unittest.TestCase):
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
