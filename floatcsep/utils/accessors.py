import time
import git
import requests
import hashlib
import os
import shutil


def from_zenodo(record_id, folder, force=False, keys=None):
    record_url = f"https://zenodo.org/api/records/{record_id}"
    max_tries = 5

    os.makedirs(folder, exist_ok=True)

    for attempt in range(1, max_tries + 1):
        r = requests.get(record_url, timeout=30, headers={"User-Agent": "floatcsep"})

        if r.status_code == 200:
            break

        if r.status_code == 403:
            text = (r.text or "").lower()
            if "unusual traffic" in text or "<html" in text:
                snippet = (r.text or "")[:400].replace("\n", "\\n")
                raise RuntimeError(
                    "Zenodo returned HTTP 403 and appears to be blocking this network/IP due "
                    "to unusual traffic.\n"
                    f"URL: {record_url}\n"
                    f"Response snippet: {snippet}"
                )
            r.raise_for_status()

        if r.status_code in (429, 500, 502, 503, 504):
            wait = min(2 ** (attempt - 1), 30)
            ra = r.headers.get("Retry-After")
            if ra:
                try:
                    wait = max(wait, int(ra))
                except ValueError:
                    pass
            time.sleep(wait)
            continue

        r.raise_for_status()
    else:
        raise RuntimeError(f"Zenodo API request failed after {max_tries} attempts: {record_url}")

    try:
        data = r.json()
    except Exception as e:
        snippet = (r.text or "")[:400].replace("\n", "\\n")
        raise RuntimeError(
            "Zenodo API did not return valid JSON.\n"
            f"URL: {record_url}\n"
            f"Content-Type: {r.headers.get('Content-Type')!r}\n"
            f"Snippet: {snippet}"
        ) from e

    files = data.get("files", [])
    if not isinstance(files, list):
        raise RuntimeError(f"Zenodo record JSON missing expected 'files' list: {record_url}")

    if keys is not None:
        wanted = set(keys)
        files = [f for f in files if f.get("key") in wanted]
        missing = wanted - {f.get("key") for f in files}
        if missing:
            raise FileNotFoundError(
                f"Zenodo record {record_id} does not contain required file(s): {sorted(missing)}"
            )

    download_urls = [f["links"]["self"] for f in files]
    filenames = [(f["key"], f["checksum"]) for f in files]

    for (fname, checksum), url in zip(filenames, download_urls):
        full_path = os.path.join(folder, fname)

        if os.path.exists(full_path):
            value, digest = check_hash(full_path, checksum)
            if value != digest:
                print(f"Checksum differs, re-downloading {fname} ...")
                download_file(url, full_path)
            elif force:
                print(f"Re-downloading {fname} ...")
                download_file(url, full_path)
            else:
                print(f"Found {fname}. Checksum OK.")
        else:
            print(f"Downloading {fname} ...")
            download_file(url, full_path)

        value, digest = check_hash(full_path, checksum)
        if value != digest:
            raise Exception("Error: Checksum does not match")


def from_git(url, path, branch=None, depth=1, force=False, **kwargs):
    kwargs = dict(kwargs, depth=depth)
    git.refresh()

    if os.path.exists(path):
        if force:
            shutil.rmtree(path)
        elif os.listdir(path):
            raise ValueError(f"Cannot clone into non-empty directory: {path}")

    os.makedirs(path, exist_ok=True)

    repo = git.Repo.clone_from(url, path, branch=branch, **kwargs)
    git_dir = os.path.join(path, ".git")
    if os.path.isdir(git_dir):
        shutil.rmtree(git_dir)

    return repo


def download_file(url: str, filename: str) -> None:
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    r = requests.get(url, timeout=30, stream=True, headers={"User-Agent": "floatcsep"})
    r.raise_for_status()

    cl = r.headers.get("Content-Length") or r.headers.get("content-length")
    try:
        total_size = int(cl) if cl else 0
    except ValueError:
        total_size = 0

    base = os.path.basename(filename)
    if total_size:
        print(f"{base} ({total_size / (1024 * 1024):.2f} MB)")
    else:
        print(f"{base}")

    with open(filename, "wb") as f:
        for data in r.iter_content(chunk_size=1024 * 64):
            if not data:
                continue
            f.write(data)

    print(f"Complete: {base}")


def check_hash(filename, checksum):
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
