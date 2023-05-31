import os.path
import vcr
from datetime import datetime
from floatcsep.accessors import query_gcmt, _query_gcmt, from_zenodo, \
    from_git, _check_hash
import unittest
from unittest import mock

root_dir = os.path.dirname(os.path.abspath(__file__))

def gcmt_dir():
    data_dir = os.path.join(root_dir, '../artifacts', 'gcmt')
    return data_dir


def zenodo_dir():
    data_dir = os.path.join(root_dir, '../artifacts', 'zenodo')
    return data_dir


class TestCatalogGetter(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.makedirs(gcmt_dir(), exist_ok=True)
        cls._fname = os.path.join(gcmt_dir(), 'test_cat')

    def test_gcmt_search(self):
        tape_file = os.path.join(gcmt_dir(), 'vcr_search.yaml')
        with vcr.use_cassette(tape_file):
            # Maule, Chile
            eventlist = \
                _query_gcmt(start_time=datetime(2010, 2, 26),
                            end_time=datetime(2010, 3, 2),
                            min_magnitude=6)
            event = eventlist[0]
            assert event[0] == '2844986'

    def test_gcmt_summary(self):
        tape_file = os.path.join(gcmt_dir(), 'vcr_summary.yaml')
        with vcr.use_cassette(tape_file):
            eventlist = \
                _query_gcmt(start_time=datetime(2010, 2, 26),
                            end_time=datetime(2010, 3, 2),
                            min_magnitude=7)
            event = eventlist[0]
            cmp = "('2844986', 1267252514000, -35.98, -73.15, 23.2, 8.8)"
            assert str(event) == cmp
            assert event[0] == '2844986'
            assert datetime.fromtimestamp(
                event[1] / 1000.) == datetime.fromtimestamp(1267252514)
            assert event[2] == -35.98
            assert event[3] == -73.15
            assert event[4] == 23.2
            assert event[5] == 8.8

    def test_catalog_query_plot(self):
        start_datetime = datetime(2020, 1, 1)
        end_datetime = datetime(2020, 3, 1)
        catalog = query_gcmt(start_time=start_datetime,
                             end_time=end_datetime,
                             min_magnitude=5.95)
        catalog.plot(set_global=True, plot_args={'filename': self._fname,
                                                 'basemap': 'stock_img'})
        assert os.path.isfile(self._fname + '.png')
        assert os.path.isfile(self._fname + '.pdf')

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            os.remove(os.path.join(gcmt_dir(), cls._fname + '.pdf'))
            os.remove(os.path.join(gcmt_dir(), cls._fname + '.png'))
        except OSError:
            pass

class TestZenodoGetter(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.makedirs(zenodo_dir(), exist_ok=True)
        cls._txt = os.path.join(zenodo_dir(), 'dummy.txt')
        cls._tar = os.path.join(zenodo_dir(), 'dummy.tar')

    def test_zenodo_query(self):
        from_zenodo(4739912, zenodo_dir())
        assert os.path.isfile(self._txt)
        assert os.path.isfile(self._tar)
        with open(self._txt, 'r') as dummy:
            assert dummy.readline() == 'test'
        _check_hash(self._tar, 'md5:17f80d606ff085751998ac4050cc614c')

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(os.path.join(zenodo_dir(), 'dummy.txt'))
        os.remove(os.path.join(zenodo_dir(), 'dummy.tar'))
        os.rmdir(zenodo_dir())


class TestGitter(unittest.TestCase):
    @mock.patch('floatcsep.accessors.git.Repo')
    @mock.patch('git.Git')
    def runTest(self, mock_git, mock_repo):
        p = mock.PropertyMock(return_value=False)
        type(mock_repo.clone_from.return_value).bare = p
        from_git(
            '/tmp/testrepo',
            'git@github.com:github/testrepo.git',
            'master'
        )
        mock_git.checkout.called_once_with('master')
