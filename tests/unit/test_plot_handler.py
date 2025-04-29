import unittest
from unittest.mock import patch, MagicMock
import floatcsep.postprocess.plot_handler as plot_handler


class TestPlotHandler(unittest.TestCase):

    @patch("matplotlib.pyplot.savefig")
    @patch("floatcsep.postprocess.plot_handler.timewindow2str")
    def test_plot_results(self, mock_timewindow2str, mock_savefig):
        mock_experiment = MagicMock()
        mock_test = MagicMock()
        mock_experiment.tests = [mock_test]
        mock_timewindow2str.return_value = ["2021-01-01", "2021-12-31"]

        plot_handler.plot_results(mock_experiment)

        mock_timewindow2str.assert_called_once_with(mock_experiment.time_windows)
        mock_test.plot_results.assert_called_once_with(
            ["2021-01-01", "2021-12-31"], mock_experiment.models, mock_experiment.registry
        )

    @patch("matplotlib.pyplot.savefig")
    @patch("floatcsep.postprocess.plot_handler.parse_plot_config")
    @patch("floatcsep.postprocess.plot_handler.parse_projection")
    def test_plot_forecasts(self, mock_parse_projection, mock_parse_plot_config, mock_savefig):
        mock_experiment = MagicMock()
        mock_model = MagicMock()
        mock_experiment.models = [mock_model]
        mock_parse_plot_config.return_value = {"projection": "Mercator"}
        mock_parse_projection.return_value = MagicMock()
        mock_experiment.postprocess.get.return_value = True

        plot_handler.plot_forecasts(mock_experiment)

        mock_parse_plot_config.assert_called_once_with(
            mock_experiment.postprocess.get("plot_forecasts", {})
        )
        mock_model.get_forecast().plot.assert_called()

        # Verify that pyplot.savefig was called to save the plot
        mock_savefig.assert_called()

    @patch("matplotlib.pyplot.Figure.savefig")  # Mocking savefig on the Figure object
    @patch("floatcsep.postprocess.plot_handler.parse_plot_config")
    @patch("floatcsep.postprocess.plot_handler.parse_projection")
    def test_plot_catalogs(
        self, mock_parse_projection, mock_parse_plot_config, mock_savefig
    ):
        # Mock the experiment and its components
        mock_experiment = MagicMock()
        mock_catalog = MagicMock()
        mock_plot = MagicMock()
        mock_ax = MagicMock()
        mock_figure = MagicMock()

        mock_experiment.catalog_repo.get_test_cat = MagicMock(return_value=mock_catalog)
        mock_catalog.plot = mock_plot
        mock_plot.return_value = mock_ax
        mock_ax.get_figure.return_value = (
            mock_figure
        )

        mock_parse_plot_config.return_value = {"projection": "Mercator"}
        mock_parse_projection.return_value = MagicMock()
        mock_experiment.registry.get_figure_key.return_value = "cat.png"

        plot_handler.plot_catalogs(mock_experiment)

        mock_parse_plot_config.assert_called_once_with(
            mock_experiment.postprocess.get("plot_catalog", {})
        )

        mock_plot.assert_called_once_with(
            plot_args=mock_parse_plot_config.return_value,
        )

        mock_figure.savefig.assert_called_once_with(
            "cat.png", dpi=300
        )

        mock_savefig.assert_called()

    @patch("os.path.isfile", return_value=True)
    @patch("os.path.realpath", return_value="dir")
    @patch("os.path.dirname", return_value="dir")
    @patch("importlib.util.spec_from_file_location")
    @patch("importlib.util.module_from_spec")
    def test_plot_custom(self, mock_module_from_spec, mock_spec_from_file_location,
                         mock_dirname, mock_realpath, mock_isfile):
        mock_experiment = MagicMock()
        mock_spec = MagicMock()
        mock_module = MagicMock()
        mock_func = MagicMock()

        mock_spec_from_file_location.return_value = mock_spec
        mock_module_from_spec.return_value = mock_module
        mock_module.plot_function = mock_func
        mock_experiment.postprocess.get.return_value = "custom_script.py:plot_function"
        mock_experiment.registry.abs.return_value = "custom_script.py"

        plot_handler.plot_custom(mock_experiment)

        mock_spec_from_file_location.assert_called_once_with("custom_script", "custom_script.py")
        mock_module_from_spec.assert_called_once_with(mock_spec)
        mock_func.assert_called_once_with(mock_experiment)

    def test_parse_plot_config(self):
        # Test True case
        result = plot_handler.parse_plot_config(True)
        self.assertEqual(result, {})

        # Test False case
        result = plot_handler.parse_plot_config(False)
        self.assertIsNone(result)

        # Test dict case
        mock_dict = {"key": "value"}
        result = plot_handler.parse_plot_config(mock_dict)
        self.assertEqual(result, mock_dict)

        # Test string case with valid script and function
        result = plot_handler.parse_plot_config("script.py:plot_func")
        self.assertEqual(result, ("script.py", "plot_func"))

    def test_parse_projection(self):
        # Test None case
        result = plot_handler.parse_projection(None)
        self.assertEqual(result.__class__.__name__, "PlateCarree")

        # Test dict case with valid projection
        mock_config = {"Mercator": {"central_longitude": 0.0}}
        result = plot_handler.parse_projection(mock_config)
        self.assertEqual(result.__class__.__name__, "Mercator")

        # Test invalid projection case
        result = plot_handler.parse_projection("InvalidProjection")
        self.assertEqual(result.__class__.__name__, "PlateCarree")


if __name__ == "__main__":
    unittest.main()
