from csep.utils.documents import MarkdownReport

""" 
Use the MarkdownReport class to create output for the experiment

1. string templates are stored for each evaluation
2. string templates are stored for each forecast
3. report should include
    - plots of catalog
    - plots of forecasts
    - evaluation results
    - metadata from run, (maybe json dump of experiment class)
"""

def generate_markdown_report(experiment_config):
    """ Generates markdown report from experiment configuration

        Use the return value of this function to write the report to disk.

        Args:
            experiment_config (config.Experiment): configuration to generate report

        Returns:
            report (str): markdown report
    """
    pass

