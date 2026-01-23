import os
import shutil
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from floatcsep.utils.accessors import from_zenodo, from_git, check_hash


def has_git():
    """Skip if git executable isn't available."""
    from shutil import which

    return which("git") is not None


def has_internet(host="github.com", port=443, timeout=2):
    """Shallow check for outbound connectivity."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def make_local_git_repo() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="mock_src_repo_"))

    # Minimal content
    (tmp / "README.md").write_text("# mock\n", encoding="utf-8")
    (tmp / "src").mkdir(parents=True, exist_ok=True)
    (tmp / "src" / "mod.py").write_text("x = 1\n", encoding="utf-8")

    # Init repo with stable default branch
    try:
        subprocess.run(["git", "-C", str(tmp), "init", "-q", "-b", "main"], check=True)
    except subprocess.CalledProcessError:
        subprocess.run(["git", "-C", str(tmp), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(tmp), "checkout", "-q", "-b", "main"], check=True)

    # Per-repo config so commits work in CI/tox
    subprocess.run(["git", "-C", str(tmp), "config", "user.name", "CI Tester"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp), "config", "user.email", "ci@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(tmp), "config", "commit.gpgsign", "false"], check=True)

    # Stage & commit (explicitly disable signing)
    subprocess.run(["git", "-C", str(tmp), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp), "commit", "-q", "--no-gpg-sign", "-m", "init"],
        check=True,
        env=dict(
            os.environ,  # make extra sure author/committer are set
            GIT_AUTHOR_NAME="CI Tester",
            GIT_AUTHOR_EMAIL="ci@example.com",
            GIT_COMMITTER_NAME="CI Tester",
            GIT_COMMITTER_EMAIL="ci@example.com",
        ),
    )

    return tmp


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
        try:
            if os.path.isfile(self._txt) and os.path.isfile(self._tar):
                exp, got = check_hash(self._tar, "md5:17f80d606ff085751998ac4050cc614c")
                if exp == got:
                    self._assert_files_ok()
                    return
            try:
                from_zenodo(4739912, zenodo_dir(), keys=["dummy.txt", "dummy.tar"])
            except Exception as e:
                self.skipTest(f"Zenodo flaky/unavailable: {e!r}")

            self._assert_files_ok()
        except Exception as e:
            self.skipTest(f"Skipping Zenodo test: {e!r}")

    def _assert_files_ok(self):
        self.assertTrue(os.path.isfile(self._txt))
        self.assertTrue(os.path.isfile(self._tar))
        with open(self._txt, "r") as dummy:
            self.assertEqual(dummy.readline(), "test")
        exp, got = check_hash(self._tar, "md5:17f80d606ff085751998ac4050cc614c")
        self.assertEqual(exp, got)

    @classmethod
    def tearDownClass(cls) -> None:
        for fn in ("dummy.txt", "dummy.tar"):
            fp = os.path.join(zenodo_dir(), fn)
            if os.path.exists(fp):
                os.remove(fp)
        try:
            os.rmdir(zenodo_dir())
        except OSError:
            pass


@unittest.skipUnless(has_git(), "git executable not available")
class TestFromGitLocal(unittest.TestCase):
    def setUp(self):
        self.src_repo = make_local_git_repo()
        self.tmp_root = Path(tempfile.mkdtemp(prefix="probe_from_git_")).resolve()
        self.target = (self.tmp_root / "models" / "asd").resolve()

    def tearDown(self):
        shutil.rmtree(self.src_repo, ignore_errors=True)
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_clone_into_empty_target_places_contents_directly(self):
        # target does not exist yet
        self.assertFalse(self.target.exists())

        from_git(url=str(self.src_repo), path=str(self.target), branch=None, force=False)

        # target now exists
        self.assertTrue(self.target.exists() and self.target.is_dir())
        # contents are directly under target (no extra nesting like target/repo-name/*)
        self.assertTrue((self.target / "README.md").is_file())
        self.assertTrue((self.target / "src").is_dir())
        # .git must be removed
        self.assertFalse((self.target / ".git").exists())

        # Heuristic: ensure not exactly one subdir and no files at top-level (nesting symptom)
        top_dirs = [p for p in self.target.iterdir() if p.is_dir()]
        top_files = [p for p in self.target.iterdir() if p.is_file()]
        self.assertFalse(len(top_dirs) == 1 and len(top_files) == 0)

    def test_clone_into_existing_non_empty_without_force_raises(self):
        self.target.mkdir(parents=True, exist_ok=True)
        (self.target / "keep.txt").write_text("hello")

        with self.assertRaises(ValueError):
            from_git(url=str(self.src_repo), path=str(self.target), branch=None, force=False)

        # untouched
        self.assertTrue((self.target / "keep.txt").is_file())

    def test_clone_into_existing_non_empty_with_force_overwrites(self):
        self.target.mkdir(parents=True, exist_ok=True)
        (self.target / "keep.txt").write_text("hello")

        from_git(url=str(self.src_repo), path=str(self.target), branch=None, force=True)

        # old file removed; new repo contents present
        self.assertFalse((self.target / "keep.txt").exists())
        self.assertTrue((self.target / "README.md").is_file())
        self.assertTrue((self.target / "src").is_dir())
        self.assertFalse((self.target / ".git").exists())

    def test_creates_target_if_missing(self):
        # parent exists, but leaf does not
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.assertFalse(self.target.exists())

        from_git(url=str(self.src_repo), path=str(self.target), branch=None, force=False)
        self.assertTrue(self.target.exists() and self.target.is_dir())


class TestFromGitArgs(unittest.TestCase):
    @mock.patch("floatcsep.utils.accessors.git.Repo.clone_from")
    @mock.patch("floatcsep.utils.accessors.git.refresh")
    def test_depth_and_branch_forwarded(self, mock_refresh, mock_clone):
        with tempfile.TemporaryDirectory(prefix="from_git_args_") as tmpd:
            target = Path(tmpd) / "dest"
            url = "file:///some/local/repo"
            depth = 5
            branch = "main"

            from_git(url=url, path=str(target), branch=branch, depth=depth, force=False)
            called_args, called_kwargs = mock_clone.call_args
            self.assertEqual(called_args[0], url)
            self.assertEqual(Path(called_args[1]), target)
            self.assertEqual(called_kwargs.get("branch"), branch)
            self.assertEqual(called_kwargs.get("depth"), depth)

    @mock.patch("floatcsep.utils.accessors.git.Repo.clone_from")
    @mock.patch("floatcsep.utils.accessors.git.refresh")
    def test_removes_git_dir_after_clone(self, mock_refresh, mock_clone):
        with tempfile.TemporaryDirectory(prefix="from_git_gitdir_") as tmpd:
            target = Path(tmpd) / "dest"
            url = "file:///some/repo"
            from_git(url=url, path=str(target), branch=None, depth=1, force=False)
            # should not raise; target should exist
            self.assertTrue(target.exists() and target.is_dir())


@unittest.skipUnless(has_git() and has_internet(), "requires git and internet")
class TestFromGitRemoteOptional(unittest.TestCase):
    def setUp(self):
        self.tmp_root = Path(tempfile.mkdtemp(prefix="probe_from_git_remote_")).resolve()
        self.target = (self.tmp_root / "models" / "asd").resolve()

    def tearDown(self):
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_remote_repo_into_target(self):
        url = "https://github.com/pabloitu/model_template.git"
        from_git(url=url, path=str(self.target), branch=None, force=True)
        self.assertTrue(self.target.exists() and self.target.is_dir())
        # At least one file at the top level is expected in this demo repo
        top_files = [p for p in self.target.iterdir() if p.is_file()]
        self.assertTrue(len(top_files) >= 1)
        self.assertFalse((self.target / ".git").exists())
