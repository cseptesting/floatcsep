import unittest
from unittest.mock import patch, MagicMock
import floatcsep.commands.main as main_module


class TestMainModule(unittest.TestCase):

    @patch('floatcsep.commands.main.Experiment')
    @patch('floatcsep.commands.main.plot_catalogs')
    @patch('floatcsep.commands.main.plot_forecasts')
    @patch('floatcsep.commands.main.plot_results')
    @patch('floatcsep.commands.main.plot_custom')
    @patch('floatcsep.commands.main.generate_report')
    def test_run(self, mock_generate_report, mock_plot_custom, mock_plot_results,
                 mock_plot_forecasts, mock_plot_catalogs, mock_experiment):
        # Mock Experiment instance and its methods
        mock_exp_instance = MagicMock()
        mock_experiment.from_yml.return_value = mock_exp_instance

        # Call the function
        main_module.run(config='dummy_config')

        # Verify the calls to the Experiment class methods
        mock_experiment.from_yml.assert_called_once_with(config_yml='dummy_config')
        mock_exp_instance.stage_models.assert_called_once()
        mock_exp_instance.set_tasks.assert_called_once()
        mock_exp_instance.run.assert_called_once()

        # Verify that plotting and report generation functions were called
        mock_plot_catalogs.assert_called_once_with(experiment=mock_exp_instance)
        mock_plot_forecasts.assert_called_once_with(experiment=mock_exp_instance)
        mock_plot_results.assert_called_once_with(experiment=mock_exp_instance)
        mock_plot_custom.assert_called_once_with(experiment=mock_exp_instance)
        mock_generate_report.assert_called_once_with(experiment=mock_exp_instance)

    @patch('floatcsep.commands.main.Experiment')
    def test_stage(self, mock_experiment):
        # Mock Experiment instance and its methods
        mock_exp_instance = MagicMock()
        mock_experiment.from_yml.return_value = mock_exp_instance

        # Call the function
        main_module.stage(config='dummy_config')

        # Verify the calls to the Experiment class methods
        mock_experiment.from_yml.assert_called_once_with(config_yml='dummy_config')
        mock_exp_instance.stage_models.assert_called_once()

    @patch('floatcsep.commands.main.Experiment')
    @patch('floatcsep.commands.main.plot_catalogs')
    @patch('floatcsep.commands.main.plot_forecasts')
    @patch('floatcsep.commands.main.plot_results')
    @patch('floatcsep.commands.main.plot_custom')
    @patch('floatcsep.commands.main.generate_report')
    def test_plot(self, mock_generate_report, mock_plot_custom, mock_plot_results,
                  mock_plot_forecasts, mock_plot_catalogs, mock_experiment):
        # Mock Experiment instance and its methods
        mock_exp_instance = MagicMock()
        mock_experiment.from_yml.return_value = mock_exp_instance

        # Call the function
        main_module.plot(config='dummy_config')

        # Verify the calls to the Experiment class methods
        mock_experiment.from_yml.assert_called_once_with(config_yml='dummy_config')
        mock_exp_instance.stage_models.assert_called_once()
        mock_exp_instance.set_tasks.assert_called_once()

        # Verify that plotting and report generation functions were called
        mock_plot_catalogs.assert_called_once_with(experiment=mock_exp_instance)
        mock_plot_forecasts.assert_called_once_with(experiment=mock_exp_instance)
        mock_plot_results.assert_called_once_with(experiment=mock_exp_instance)
        mock_plot_custom.assert_called_once_with(experiment=mock_exp_instance)
        mock_generate_report.assert_called_once_with(experiment=mock_exp_instance)

    @patch('floatcsep.commands.main.Experiment')
    @patch('floatcsep.commands.main.ExperimentComparison')
    @patch('floatcsep.commands.main.reproducibility_report')
    def test_reproduce(self, mock_reproducibility_report, mock_exp_comparison, mock_experiment):
        # Mock Experiment instances and methods
        mock_reproduced_exp = MagicMock()
        mock_original_exp = MagicMock()
        mock_experiment.from_yml.side_effect = [mock_reproduced_exp, mock_original_exp]

        mock_comp_instance = MagicMock()
        mock_exp_comparison.return_value = mock_comp_instance

        # Call the function
        main_module.reproduce(config='dummy_config')

        # Verify the calls to the Experiment class methods
        mock_experiment.from_yml.assert_any_call('dummy_config', repr_dir="reproduced")
        mock_reproduced_exp.stage_models.assert_called_once()
        mock_reproduced_exp.set_tasks.assert_called_once()
        mock_reproduced_exp.run.assert_called_once()

        mock_experiment.from_yml.assert_any_call(mock_reproduced_exp.original_config,
                                                 rundir=mock_reproduced_exp.original_run_dir)
        mock_original_exp.stage_models.assert_called_once()
        mock_original_exp.set_tasks.assert_called_once()

        # Verify comparison and reproducibility report calls
        mock_exp_comparison.assert_called_once_with(mock_original_exp, mock_reproduced_exp)
        mock_comp_instance.compare_results.assert_called_once()
        mock_reproducibility_report.assert_called_once_with(exp_comparison=mock_comp_instance)


if __name__ == '__main__':
    unittest.main()
