import os.path
from floatcsep.utils.accessors import from_zenodo, from_git, check_hash
import unittest
from unittest import mock

root_dir = os.path.dirname(os.path.abspath(__file__))


def zenodo_dir():
    data_dir = os.path.join(root_dir, "../artifacts", "zenodo")
    return data_dir


class TestZenodoGetter(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.makedirs(zenodo_dir(), exist_ok=True)
        cls._txt = os.path.join(zenodo_dir(), "dummy.txt")
        cls._tar = os.path.join(zenodo_dir(), "dummy.tar")

    def test_zenodo_query(self):
        from_zenodo(4739912, zenodo_dir())
        assert os.path.isfile(self._txt)
        assert os.path.isfile(self._tar)
        with open(self._txt, "r") as dummy:
            assert dummy.readline() == "test"
        check_hash(self._tar, "md5:17f80d606ff085751998ac4050cc614c")

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(os.path.join(zenodo_dir(), "dummy.txt"))
        os.remove(os.path.join(zenodo_dir(), "dummy.tar"))
        os.rmdir(zenodo_dir())


class TestGitter(unittest.TestCase):
    @mock.patch("floatcsep.utils.accessors.git.Repo")
    @mock.patch("git.Git")
    def runTest(self, mock_git, mock_repo):
        p = mock.PropertyMock(return_value=False)
        type(mock_repo.clone_from.return_value).bare = p
        from_git("/tmp/testrepo", "git@github.com:github/testrepo.git", "master")
        mock_git.checkout.called_once_with("master")
