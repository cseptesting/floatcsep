"""

    Overall experiment steps:
        0. Setup directory structure for run-time
            - Results folder
            - Observations folder
            - Forecasts folder
            - Figures folder
            - README.md goes in top-level
            - Use test datetime for folder name

        1. Retrieve data from online repository (Zenodo and ISC)

            - Use experiment config class to determine the filepath of these models. If not found download, else skip
              downloading.
            - Download gCMT catalog from ISC. If catalog is found (the run is being re-created), then skip downloading and
              filtering.
                - Filter catalog according to experiment definition
                - Write ASCII version of catalog
                - Update experiment manifest with filepath

        2. Prepare forecast files from models

            - Using same logic, only prepare these files if they are not found locally (ie, new run of the experiment)
            - The catalogs should be filtered to the same time horizon and region from experiment

        3. Evaluate models

            - Run the evaluations using the information stored in the experiment config
            - Update experiment class with information from evaluation runs

        4. Clean-up steps

            - Prepare Markdown report using the Experiment class
            - Commit run results to GitLab (if running from authorized user)
"""
import os
import datetime
from collections import defaultdict
from gefe.config import exp

def run(test_date_string):
    # parse test date from date string
    try:
        test_date = datetime.datetime.strptime(test_date_string, '%Y-%m-%dT%H:%M:%S')
    except:
        raise RuntimeError("Error parsing test date string. Should have format='%Y-%m-%dT%H:%M:%S")
    exp.set_test_date(test_date)

    print('Experiment is configured with the following parameters')
    print('======================================================\n')
    print(exp.to_yaml())

    # 0. create expected paths based on experiment configuration
    exp.get_run_struct()

    # 1. retrieve data from isc catalog reader or from previous run
    # todo: download models from zenodo instead of gitlab
    catalog = exp.get_catalog()
    if not os.path.exists(exp.target_paths['catalog']):
        catalog.write_json(exp.target_paths['catalog'])

    # 2. Prepare each test that needs to be computed, could be oom problem for large sized tests
    test_list = exp.prepare_all_tests()

    # 3. Evaluate models
    run_results = defaultdict(list)
    for test in test_list:
        result = exp.run_test(test)
        run_results[test.name].append(result)

    # 4. Plot results
    exp.plot_results(run_results)

    # 4.1 Generate report
    exp.generate_report()
