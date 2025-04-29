import os.path
import tempfile
from unittest.mock import patch

import numpy
from unittest import TestCase
from datetime import datetime
from floatcsep.experiment import Experiment
from csep.core import poisson_evaluations

_dir = os.path.dirname(__file__)
_model_cfg = os.path.normpath(os.path.join(_dir, "../artifacts", "models", "model_cfg.yml"))
_region = os.path.normpath(os.path.join(_dir, "../artifacts", "regions", "mock_region"))
_time_config = {"start_date": datetime(2021, 1, 1), "end_date": datetime(2022, 1, 1)}
_region_config = {
    "region": _region,
    "mag_max": 10.0,
    "mag_min": 1.0,
    "mag_bin": 0.1,
    "depth_min": 0,
    "depth_max": 1,
}
_cat = os.path.normpath(os.path.join(_dir, "../artifacts", "catalog.json"))


class TestExperiment(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self):
        self.makedirs_patch = patch("os.makedirs", autospec=True)
        self.mock_makedirs = self.makedirs_patch.start()

    def tearDown(self):
        self.makedirs_patch.stop()

    def assertEqualExperiment(self, exp_a, exp_b):
        self.assertEqual(exp_a.name, exp_b.name)
        self.assertEqual(exp_a.registry.workdir, os.getcwd())
        self.assertEqual(exp_a.registry.workdir, exp_b.registry.workdir)
        self.assertEqual(exp_a.start_date, exp_b.start_date)
        print(exp_a.time_windows, exp_b.time_windows)
        self.assertEqual(exp_a.time_windows, exp_b.time_windows)
        self.assertEqual(exp_a.exp_class, exp_b.exp_class)
        self.assertEqual(exp_a.region, exp_b.region)
        numpy.testing.assert_equal(exp_a.magnitudes, exp_b.magnitudes)
        numpy.testing.assert_equal(exp_a.depths, exp_b.depths)
        self.assertEqual(exp_a.catalog, exp_b.catalog)

    def test_init(self):
        exp_a = Experiment(**_time_config, **_region_config, catalog=_cat)
        exp_b = Experiment(time_config=_time_config, region_config=_region_config, catalog=_cat)
        self.assertEqualExperiment(exp_a, exp_b)

    def test_to_dict(self):
        time_config = {
            "start_date": datetime(2020, 1, 1),
            "end_date": datetime(2021, 1, 1),
            "horizon": "6 month",
            "growth": "cumulative",
        }

        region_config = {
            "region": "california_relm_region",
            "mag_max": 9.0,
            "mag_min": 3.0,
            "mag_bin": 0.1,
            "depth_min": -2,
            "depth_max": 70,
        }

        exp_a = Experiment(name="test", **time_config, **region_config, catalog=_cat)
        dict_ = {
            "name": "test",
            "path": os.getcwd(),
            "run_dir": "results",
            "config_file": None,
            "models": [],
            "tests": [],
            "time_config": {
                "exp_class": "ti",
                "start_date": datetime(2020, 1, 1),
                "end_date": datetime(2021, 1, 1),
                "horizon": "6-months",
                "growth": "cumulative",
            },
            "region_config": {
                "region": "california_relm_region",
                "mag_max": 9.0,
                "mag_min": 3.0,
                "mag_bin": 0.1,
                "depth_min": -2,
                "depth_max": 70,
            },
            "catalog": os.path.relpath(_cat, os.getcwd()),
        }
        self.assertEqual(dict_, exp_a.as_dict())

    def test_to_yml(self):
        time_config = {
            "start_date": datetime(2021, 1, 1),
            "end_date": datetime(2022, 1, 1),
            "intervals": 12,
        }

        region_config = {
            "region": "california_relm_region",
            "mag_max": 9.0,
            "mag_min": 3.0,
            "mag_bin": 0.1,
            "depth_min": -2,
            "depth_max": 70,
        }

        exp_a = Experiment(**time_config, **region_config, catalog=_cat)
        file_ = tempfile.mkstemp()[1]
        exp_a.to_yml(file_)
        exp_b = Experiment.from_yml(file_)

        self.assertEqualExperiment(exp_a, exp_b)

        file_ = tempfile.mkstemp()[1]
        exp_a.to_yml(file_)
        exp_c = Experiment.from_yml(file_)
        self.assertEqualExperiment(exp_a, exp_c)

    def test_set_models(self):
        exp = Experiment(
            **_time_config, **_region_config, model_config=_model_cfg, catalog=_cat
        )
        names = [i.name for i in exp.models]
        self.assertEqual(["mock", "qtree@team10", "qtree@team25"], names)
        m1_path = os.path.normpath(
            os.path.join(_dir, "../artifacts", "models", "qtree", "TEAM=N10L11.csv")
        )

    def test_stage_models(self):
        exp = Experiment(
            **_time_config, **_region_config, model_config=_model_cfg, catalog=_cat
        )
        exp.stage_models()

        dbpath = os.path.relpath(os.path.join(_dir, "../artifacts", "models", "model.hdf5"))
        self.assertEqual(exp.models[0].registry.database, dbpath)

    def test_set_tests(self):
        test_cfg = os.path.normpath(
            os.path.join(_dir, "../artifacts", "evaluations", "tests_cfg.yml")
        )
        exp = Experiment(**_time_config, **_region_config, test_config=test_cfg, catalog=_cat)

        funcs = [i.func for i in exp.tests]
        funcs_expected = [
            poisson_evaluations.number_test,
            poisson_evaluations.spatial_test,
            poisson_evaluations.paired_t_test,
        ]
        for i, j in zip(funcs, funcs_expected):
            self.assertIs(i, j)

    def test_prepare_subcatalog(self):
        time_config = {**_time_config}
        exp = Experiment(**time_config, **_region_config, catalog=_cat)
        tstring = "2020-08-01_2021-01-02"

        with tempfile.NamedTemporaryFile() as file_:

            def filetree(*args):
                return file_.name

            exp.path = filetree
            # with patch.object(exp, 'filetree', filetree):
            #     print(file_.name)
            #     exp.set_test_cat(tstring)
            #     cat = CSEPCatalog.load_json(file_.name)
            #     numpy.testing.assert_equal(1609455600000, cat.data[0][1])

    @classmethod
    def tearDownClass(cls) -> None:
        path_ = os.path.join(_dir, "../artifacts", "models", "model.hdf5")
        if os.path.isfile(path_):
            os.remove(path_)
