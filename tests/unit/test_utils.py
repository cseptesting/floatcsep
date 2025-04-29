import os.path
import unittest
import numpy
from datetime import datetime

import csep
import csep.utils.plots
import csep.core.regions

import floatcsep
import floatcsep.utils.accessors
from floatcsep.utils.helpers import (
    parse_timedelta_string,
    time_windows_ti,
    read_time_cfg,
    read_region_cfg,
    parse_csep_func,
    time_windows_td,
)

root_dir = os.path.dirname(os.path.abspath(__file__))


class CsepFunctionTest(unittest.TestCase):

    def test_parse_csep_func(self):
        self.assertIsInstance(
            parse_csep_func("load_gridded_forecast"), csep.load_gridded_forecast.__class__
        )
        self.assertIsInstance(
            parse_csep_func("join_struct_arrays"), csep.utils.join_struct_arrays.__class__
        )
        self.assertIsInstance(
            parse_csep_func("plot_poisson_consistency_test"),
            csep.utils.plots.plot_poisson_consistency_test.__class__,
        )
        self.assertIsInstance(
            parse_csep_func("italy_csep_region"), csep.core.regions.italy_csep_region.__class__
        )
        self.assertIsInstance(
            parse_csep_func("from_zenodo"),
            floatcsep.utils.accessors.from_zenodo.__class__,
        )
        self.assertRaises(AttributeError, parse_csep_func, "panic_button")


class TimeUtilsTest(unittest.TestCase):

    def test_parse_time_window(self):
        dt = "1Year"
        self.assertEqual(parse_timedelta_string(dt), "1-years")
        dt = "7-Days"
        self.assertEqual(parse_timedelta_string(dt), "7-days")
        dt = "1- mOnThS"
        self.assertEqual(parse_timedelta_string(dt), "1-months")
        dt = "20        days"
        self.assertEqual(parse_timedelta_string(dt), "20-days")
        dt = "1decade"
        self.assertRaises(ValueError, parse_timedelta_string, dt)

    def test_time_windows_ti(self):
        start = datetime(2014, 1, 1)
        end = datetime(2022, 1, 1)

        self.assertEqual(time_windows_ti(start_date=start, end_date=end), [(start, end)])

        t1 = [
            (datetime(2014, 1, 1), datetime(2018, 1, 1)),
            (datetime(2018, 1, 1), datetime(2022, 1, 1)),
        ]
        self.assertEqual(time_windows_ti(start_date=start, end_date=end, intervals=2), t1)
        self.assertEqual(time_windows_ti(start_date=start, end_date=end, horizon="4-years"), t1)
        self.assertEqual(time_windows_ti(start_date=start, intervals=2, horizon="4-years"), t1)

        t2 = [
            (datetime(2014, 1, 1, 0, 0), datetime(2015, 2, 22, 10, 17, 8, 571428)),
            (
                datetime(2015, 2, 22, 10, 17, 8, 571428),
                datetime(2016, 4, 14, 20, 34, 17, 142857),
            ),
            (
                datetime(2016, 4, 14, 20, 34, 17, 142857),
                datetime(2017, 6, 6, 6, 51, 25, 714285),
            ),
            (datetime(2017, 6, 6, 6, 51, 25, 714285), datetime(2018, 7, 28, 17, 8, 34, 285714)),
            (
                datetime(2018, 7, 28, 17, 8, 34, 285714),
                datetime(2019, 9, 19, 3, 25, 42, 857142),
            ),
            (
                datetime(2019, 9, 19, 3, 25, 42, 857142),
                datetime(2020, 11, 9, 13, 42, 51, 428571),
            ),
            (datetime(2020, 11, 9, 13, 42, 51, 428571), datetime(2022, 1, 1, 0, 0)),
        ]
        self.assertEqual(time_windows_ti(start_date=start, end_date=end, intervals=7), t2)

    def test_time_windows_td(self):
        start = datetime(2010, 1, 1)
        end = datetime(2020, 1, 1)

        self.assertEqual(
            time_windows_td(start_date=start, end_date=end, timeintervals=1), [(start, end)]
        )

        self.assertEqual(
            time_windows_td(start_date=start, end_date=end, timeintervals=5),
            [
                (datetime(2010, 1, 1, 0, 0), datetime(2012, 1, 1, 9, 36)),
                (datetime(2012, 1, 1, 9, 36), datetime(2013, 12, 31, 19, 12)),
                (datetime(2013, 12, 31, 19, 12), datetime(2016, 1, 1, 4, 48)),
                (datetime(2016, 1, 1, 4, 48), datetime(2017, 12, 31, 14, 24)),
                (datetime(2017, 12, 31, 14, 24), datetime(2020, 1, 1, 0, 0)),
            ],
        )

        self.assertEqual(
            time_windows_td(start_date=start, timeintervals=1, timehorizon="1-years"),
            [(datetime(2010, 1, 1, 0, 0), datetime(2011, 1, 1, 0, 0))],
        )
        self.assertEqual(
            time_windows_td(start_date=start, timeintervals=5, timehorizon="5-days"),
            [
                (datetime(2010, 1, 1, 0, 0), datetime(2010, 1, 6, 0, 0)),
                (datetime(2010, 1, 6, 0, 0), datetime(2010, 1, 11, 0, 0)),
                (datetime(2010, 1, 11, 0, 0), datetime(2010, 1, 16, 0, 0)),
                (datetime(2010, 1, 16, 0, 0), datetime(2010, 1, 21, 0, 0)),
                (datetime(2010, 1, 21, 0, 0), datetime(2010, 1, 26, 0, 0)),
            ],
        )
        self.assertEqual(
            time_windows_td(
                start_date=start, end_date=end, timehorizon="10-years", timeoffset="10-years"
            ),
            [(datetime(2010, 1, 1, 0, 0), datetime(2020, 1, 1, 0, 0))],
        )
        self.assertEqual(
            time_windows_td(
                start_date=start, end_date=end, timehorizon="12-years", timeoffset="10-years"
            ),
            [(datetime(2010, 1, 1, 0, 0), datetime(2022, 1, 1, 0, 0))],
        )
        self.assertEqual(
            time_windows_td(
                start_date=start, end_date=end, timehorizon="5-years", timeoffset="5-years"
            ),
            [
                (datetime(2010, 1, 1, 0, 0), datetime(2015, 1, 1, 0, 0)),
                (datetime(2015, 1, 1, 0, 0), datetime(2020, 1, 1, 0, 0)),
            ],
        )
        self.assertEqual(
            time_windows_td(
                start_date=start, end_date=end, timehorizon="5-years", timeoffset="3-years"
            ),
            [
                (datetime(2010, 1, 1, 0, 0), datetime(2015, 1, 1, 0, 0)),
                (datetime(2013, 1, 1, 0, 0), datetime(2018, 1, 1, 0, 0)),
                (datetime(2016, 1, 1, 0, 0), datetime(2021, 1, 1, 0, 0)),
            ],
        )
        self.assertEqual(
            time_windows_td(
                start_date=start,
                end_date=datetime(2010, 2, 1),
                timehorizon="14-days",
                timeoffset="7-days",
            ),
            [
                (datetime(2010, 1, 1, 0, 0), datetime(2010, 1, 15, 0, 0)),
                (datetime(2010, 1, 8, 0, 0), datetime(2010, 1, 22, 0, 0)),
                (datetime(2010, 1, 15, 0, 0), datetime(2010, 1, 29, 0, 0)),
                (datetime(2010, 1, 22, 0, 0), datetime(2010, 2, 5, 0, 0)),
            ],
        )
        self.assertEqual(
            time_windows_td(
                start_date=start,
                timeintervals=3,
                timehorizon="3-years",
                timeoffset="1-years",
            ),
            [
                (datetime(2010, 1, 1, 0, 0), datetime(2013, 1, 1, 0, 0)),
                (datetime(2011, 1, 1, 0, 0), datetime(2014, 1, 1, 0, 0)),
                (datetime(2012, 1, 1, 0, 0), datetime(2015, 1, 1, 0, 0)),
            ],
        )

    def test_read_time_config(self):
        start = datetime(2014, 1, 1)
        end = datetime(2022, 1, 1)
        intervals = 2
        config = {"start_date": start, "end_date": end, "intervals": intervals}
        self.assertEqual(read_time_cfg(None, **config), read_time_cfg(config))

        short_config = {"start_date": start, "end_date": end}
        time_config = {"intervals": 2}
        full_config = {"start_date": start, "end_date": end, "intervals": 2}
        self.assertEqual(
            read_time_cfg(time_config, **short_config), read_time_cfg(None, **full_config)
        )


class RegionUtilsTest(unittest.TestCase):

    def test_magnitudes_depth(self):
        magnitudes = numpy.array([1, 1.1, 1.2])
        mag_min = 1
        mag_max = 1.2
        mag_bin = 0.1
        depth_max = 1.0
        depth_min = 0.0

        config = {
            "mag_min": mag_min,
            "mag_max": mag_max,
            "mag_bin": mag_bin,
            "depth_min": depth_min,
            "depth_max": depth_max,
        }

        region_config = read_region_cfg(config)
        self.assertEqual(8, len(region_config))
        numpy.testing.assert_equal(magnitudes, region_config["magnitudes"])
        numpy.testing.assert_equal(numpy.array([depth_min, depth_max]), region_config["depths"])

    def test_region(self):
        region_origins = numpy.array([[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]])
        region_path = os.path.join(
            os.path.dirname(__file__), "../artifacts", "regions", "mock_region"
        )
        config = {
            "region": region_path,
            "mag_min": 1,
            "mag_max": 1.2,
            "mag_bin": 0.1,
            "depth_min": 0,
            "depth_max": 1,
        }

        region_config = read_region_cfg(config)
        self.assertEqual(9, len(region_config))
        numpy.testing.assert_equal(region_origins, region_config["region"].origins())

        config = {
            "region": "italy_csep_region",
            "mag_min": 1,
            "mag_max": 1.2,
            "mag_bin": 0.1,
            "depth_min": 0,
            "depth_max": 1,
        }
        region_path = os.path.join(
            os.path.dirname(__file__), "../artifacts", "regions", "italy_midpoints"
        )
        midpoints = numpy.genfromtxt(region_path)
        region_config = read_region_cfg(config)
        numpy.testing.assert_almost_equal(midpoints, region_config["region"].midpoints())
