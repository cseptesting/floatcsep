from fecsep.accessors import from_zenodo, _check_hash
import os.path


def test_query():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    zen_path = os.path.join(root_dir, 'artifacts', 'Zenodo')
    from_zenodo(4739912, zen_path)
    txt_file = os.path.join(zen_path, 'dummy.txt')
    tar_file = os.path.join(zen_path, 'dummy.txt')
    assert os.path.isfile(txt_file)
    assert os.path.isfile(tar_file)


def test_summary():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(root_dir, 'artifacts', 'Zenodo', 'dummy.txt')
    with open(os.path.join(txt_path), 'r') as dummy:
        assert dummy.readline() == 'test'
    tar_path = os.path.join(root_dir, 'artifacts', 'Zenodo', 'dummy.tar')
    _check_hash(tar_path, 'md5:17f80d606ff085751998ac4050cc614c')
