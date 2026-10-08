import os.path
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy
from unittest import TestCase
from datetime import datetime
from floatcsep.experiment import Experiment
from floatcsep.model import TimeDependentModel
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
        self.assertEqual(exp_a.registry.workdir, Path(os.getcwd()))
        self.assertEqual(exp_a.registry.workdir, Path(exp_b.registry.workdir))
        self.assertEqual(exp_a.start_date, exp_b.start_date)
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
            "run_dir": os.path.relpath("results", os.getcwd()),
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

    def test_input_config(self):
        input_config = {
            "catalog": {
                "region": "italy_csep_collection_region",
                "mag_min": 2.5,
                "depth_max": 30,
                "start_date": datetime(2005, 1, 1),
            }
        }
        exp = Experiment(
            name="test",
            **_time_config,
            **_region_config,
            input_config=input_config,
            catalog=_cat,
        )
        self.assertEqual(2.5, exp.catalog_repo.input_cat_config["mag_min"])
        self.assertEqual(10.0, exp.catalog_repo.input_cat_config["mag_max"])
        self.assertEqual(
            "italy_csep_collection_region", exp.input_config["catalog"]["region"].name
        )
        # the testing settings are untouched
        self.assertEqual(1.0, exp.mag_min)

        dict_ = exp.as_dict()
        self.assertEqual(
            {
                "catalog": {
                    "region": "italy_csep_collection_region",
                    "mag_min": 2.5,
                    "mag_max": 10.0,
                    "depth_max": 30,
                    "start_date": datetime(2005, 1, 1),
                }
            },
            dict_["input_config"],
        )
        keys = list(dict_)
        self.assertLess(keys.index("region_config"), keys.index("input_config"))
        exp_b = Experiment(**{**dict_, "path": os.getcwd()})
        self.assertEqual(
            exp.catalog_repo.input_cat_config["mag_min"],
            exp_b.catalog_repo.input_cat_config["mag_min"],
        )

        exp_c = Experiment(name="test", **_time_config, **_region_config, catalog=_cat)
        self.assertEqual({}, exp_c.input_config)
        self.assertNotIn("input_config", exp_c.as_dict())

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

        self.assertEqual(
            exp.models[0].registry.path.resolve(),
            Path(f"{_dir}/../artifacts/models/model.csv").resolve(),
        )

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

    @patch("floatcsep.experiment.log_results_tree")
    @patch("floatcsep.experiment.log_models_tree")
    @patch.object(Experiment, "set_tree")
    def test_set_tasks_td_order_and_dependencies(self, *args):
        exp = Experiment(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 3),
            horizon="1days",
            exp_class="td",
            **_region_config,
            catalog=_cat,
        )
        models = []
        for name in ("a", "b"):
            m = MagicMock(spec=TimeDependentModel)
            m.name = name
            models.append(m)
        exp.models = models
        exp.set_tasks()

        tasks = list(exp.task_graph.tasks)
        order = [(t.method, t.kwargs["tstring"], getattr(t.obj, "name", None)) for t in tasks]
        w1, w2 = "2020-01-01_2020-01-02", "2020-01-02_2020-01-03"
        cat = exp.catalog_repo.name
        self.assertEqual(
            [
                ("set_test_cats", w1, cat),
                ("set_input_cats", w1, cat),
                ("create_forecast", w1, "a"),
                ("create_forecast", w1, "b"),
                ("set_test_cats", w2, cat),
                ("set_input_cats", w2, cat),
                ("create_forecast", w2, "a"),
                ("create_forecast", w2, "b"),
            ],
            order,
        )
        for task, deps in exp.task_graph.tasks.items():
            if task.method != "create_forecast":
                continue
            win = task.kwargs["tstring"]
            self.assertEqual(
                {("set_test_cats", win), ("set_input_cats", win)},
                {(d.method, d.kwargs["tstring"]) for d in deps},
            )

    @classmethod
    def tearDownClass(cls) -> None:
        path_ = os.path.join(_dir, "../artifacts", "models", "model.hdf5")
        if os.path.isfile(path_):
            os.remove(path_)
