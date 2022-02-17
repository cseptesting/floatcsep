""" The main experiment code will go here. 
    
    Overall experiment steps:
        0. Setup directory structure for run-time 
            - Forecasts folder
            - Results folder
            - Observations folder
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

        3. Evaluate forecasts

            - Run the evaluations using the information stored in the experiment config
            - Update experiment class with information from evaluation runs

        4. Clean-up steps
            
            - Prepare Markdown report using the Experiment class 
            - Commit run results to GitLab (if running from authorized user)
"""

