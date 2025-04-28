import git
import requests
import hashlib
import os
import sys
import shutil

def from_zenodo(record_id, folder, force=False):
    """
    Download data from a Zenodo repository.

    Downloads if file does not exist, checksum has changed in local respect to url or force

    Args:
        record_id: corresponding to the Zenodo repository
        folder: where the repository files will be downloaded
        force: force download even if file exists and checksum passes

    Returns:
    """
    # Grab the urls and filenames and checksums
    r = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=3)
    download_urls = [f["links"]["self"] for f in r.json()["files"]]
    filenames = [(f["key"], f["checksum"]) for f in r.json()["files"]]

    # Download and verify checksums
    for (fname, checksum), url in zip(filenames, download_urls):
        full_path = os.path.join(folder, fname)
        if os.path.exists(full_path):
            value, digest = check_hash(full_path, checksum)
            if value != digest:
                print(f"Checksum is different: re-downloading {fname}" f" from Zenodo...")
                download_file(url, full_path)
            elif force:
                print(f"Re-downloading {fname} from Zenodo...")
                download_file(url, full_path)
            else:
                print(f"Found file {fname}. Checksum OK.")

        else:
            print(f"Downloading {fname} from Zenodo...")
            download_file(url, full_path)
        value, digest = check_hash(full_path, checksum)
        if value != digest:
            print("Error: Checksum does not match")
            sys.exit(-1)


def from_git(url, path, branch=None, depth=1, force=False, **kwargs):
    """
    Clones a shallow repository from a git url.

    Args:
        url (str): url of the repository
        path (str): path/folder where to clone the repo
        branch (str): repository's branch to clone (default: main)
        depth (int): depth history of commits
        force (bool): If True, deletes existing path before cloning
        **kwargs: keyword args passed to Repo.clone_from

    Returns:
        the pygit repository
    """

    kwargs.update({"depth": depth})
    git.refresh()

    if os.path.exists(path):
        if force:
            shutil.rmtree(path)
        elif os.listdir(path):
            raise ValueError(f"Cannot clone into non-empty directory: {path}")
    os.makedirs(path, exist_ok=True)

    try:
        repo = git.Repo(path)
    except (git.NoSuchPathError, git.InvalidGitRepositoryError):
        repo = git.Repo.clone_from(url, path, branch=branch, **kwargs)
        git_dir = os.path.join(path, ".git")
        if os.path.isdir(git_dir):
            shutil.rmtree(git_dir)

    return repo


def download_file(url: str, filename: str) -> None:
    """
    Downloads files (from zenodo).

    Args:
        url (str): the url where the file is located
        filename (str): the filename required.
    """
    progress_bar_length = 72
    block_size = 1024

    r = requests.get(url, timeout=3, stream=True)
    total_size = r.headers.get("content-length", False)
    if not total_size:
        with requests.head(url) as h:
            try:
                total_size = int(h.headers.get("Content-Length", 0))
            except TypeError:
                total_size = 0
    else:
        total_size = int(total_size)
    download_size = 0
    if total_size:
        print(f"Downloading file with size of {total_size / block_size:.3f} kB")
    else:
        print("Downloading file with unknown size")
    with open(filename, "wb") as f:
        for data in r.iter_content(chunk_size=block_size):
            download_size += len(data)
            f.write(data)
            if total_size:
                progress = int(progress_bar_length * download_size / total_size)
                sys.stdout.write(
                    "\r[{}{}] {:.1f}%".format(
                        "█" * progress,
                        "." * (progress_bar_length - progress),
                        100 * download_size / total_size,
                    )
                )
                sys.stdout.flush()
        sys.stdout.write("\n")


def check_hash(filename, checksum):
    """Checks if existing file hash matches checksum from url."""
    algorithm, value = checksum.split(":")
    if not os.path.exists(filename):
        return value, "invalid"
    h = hashlib.new(algorithm)
    with open(filename, "rb") as f:
        while True:
            data = f.read(4096)
            if not data:
                break
            h.update(data)
    digest = h.hexdigest()
    return value, digest
