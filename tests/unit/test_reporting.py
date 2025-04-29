import unittest
from unittest.mock import patch, MagicMock
import floatcsep.postprocess.reporting as reporting


class TestReporting(unittest.TestCase):

    @patch("floatcsep.postprocess.reporting.custom_report")
    @patch("floatcsep.postprocess.reporting.MarkdownReport")
    def test_generate_report_with_custom_function(
        self, mock_markdown_report, mock_custom_report
    ):
        # Mock experiment with a custom report function
        mock_experiment = MagicMock()
        mock_experiment.postprocess.get.return_value = "custom_report_function"

        # Call the generate_report function
        reporting.generate_report(mock_experiment)

        # Assert that custom_report was called with the experiment
        mock_custom_report.assert_called_once_with("custom_report_function", mock_experiment)

    @patch("floatcsep.postprocess.reporting.MarkdownReport")
    def test_generate_standard_report(self, mock_markdown_report):
        # Mock experiment without a custom report function
        mock_experiment = MagicMock()
        mock_experiment.postprocess.get.return_value = None
        mock_experiment.registry.get_figure_key.return_value = "figure_path"
        mock_experiment.magnitudes = [0, 1]
        # Call the generate_report function
        reporting.generate_report(mock_experiment)

        # Ensure the MarkdownReport methods are called
        mock_instance = mock_markdown_report.return_value
        mock_instance.add_title.assert_called_once()
        mock_instance.add_heading.assert_called()
        mock_instance.add_figure.assert_called()


class TestMarkdownReport(unittest.TestCase):

    def test_add_title(self):
        report = reporting.MarkdownReport()
        report.add_title("Test Title", "Subtitle")
        self.assertIn("# Test Title", report.markdown[0])

    def test_add_table_of_contents(self):
        report = reporting.MarkdownReport()
        report.toc = [("Title", 1, "locator")]
        report.table_of_contents()
        self.assertIn("# Table of Contents", report.markdown[0])

    def test_save_report(self):
        report = reporting.MarkdownReport()
        report.markdown = [["# Test Title\n", "Some content\n"]]
        with patch("builtins.open", unittest.mock.mock_open()) as mock_file:
            report.save("/path/to/save")
            mock_file.assert_called_with("/path/to/save/report.md", "w")
            mock_file().writelines.assert_called_with(["# Test Title\n", "Some content\n"])



if __name__ == "__main__":
    unittest.main()
