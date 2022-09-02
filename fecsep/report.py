"""
Use the MarkdownReport class to create output for the gefe

1. string templates are stored for each evaluation
2. string templates are stored for each forecast
3. report should include
    - plots of catalog
    - plots of forecasts
    - evaluation results
    - metadata from run, (maybe json dump of gefe class)
"""
# def generate_markdown_report(mdFile, result_figs_paths, experiment_configuration):
#     """ Generates markdown report from gefe configuration
#
#         Use the return value of this function to write the report to disk.
#
#         Args:
#             mdFile : Object of MdUtils that needs to populated
#             experiment_config (config.Experiment): configuration to generate report
#
#         Returns:
#             report (str): mdFile: Object of MdUtils populated with Markdown for writing into markdown file.
#     """
#     mdFile.new_header(level=1, title='Overview')  # style is set 'atx' format by default.
#
#     mdFile.new_paragraph()
#
#     mdFile.new_paragraph("In addition to the regional experiments, CSEP promotes earthquake predictability research at a global scale "
#                      "(Eberhard et al. 2012; Taroni et al. 2014; Michael and Werner, 2018; Schorlemmer et al. 2018)."
#                      "Compared to regional approaches, global seismicity models offer great testability due to the relatively "
#                      "frequent occurrence of large events worldwide (Bayona et al. 2020). In particular, global M5.8+ "
#                      "earthquake forecasting models can be reliably ranked after only one year of prospective testing (Bird "
#                      "et al. 2015). In this regard, Eberhard et al. (2012) took a major step toward conducting a global forecast "
#                      "gefe by prospectively testing three earthquake forecasting models for the western Pacific "
#                      "region. Based on two years of testing, the authors found that a smoothed seismicity model performs "
#                      "the best, and provided useful recommendations for future global experiments. Also based on two "
#                      "years of independent observations, Strader et al. (2018) determined that the global hybrid GEAR1 "
#                      "model developed by Bird et al. (2015) significantly outperformed both of its individual model "
#                      "components, providing preliminary evidence that the combination of smoothed seismicity data and "
#                      "interseismic strain rates is suitable for global earthquake forecasting.")
#
#
#
#
#     mdFile.new_paragraph()
#
#     # Available Features
#     mdFile.new_header(level=2, title="Objectives")
#
#     objs = ["Describe the predictive skills of posited hypothesis about seismogenesis with earthquakes of M5.95+ independent observations around the globe.",
#         "Identify the methods and geophysical datasets that lead to the highest information gains in global earthquake forecasting.",
#         "Test earthquake forecast models on different grid settings.",
#         "Use Quadtree based grid to represent and evaluate earthquake forecasts."]
#     mdFile.new_list(items = objs, marked_with='-')
#
#     #mdFile.new_line("Describe the predictive skills of posited hypothesis about seismogenesis with earthquakes "
#     #                "of M5.95+ independent observations around the globe.")
#     #mdFile.new_line("Identify the methods and geophysical datasets that lead to the highest information gains in"
#     #                "global earthquake forecasting.")
#     #mdFile.new_line("Test earthquake forecast models on different grid settings.")
#     #mdFile.new_line("Use Quadtree based grid to represent and evaluate earthquake forecasts.")
#
#     mdFile.new_header(level=1, title='Forecast Experiment')
#
#     mdFile.new_header(level=2, title='Evaluation Data')
#     data = ["The authoritative evaluation data is the full Global CMT catalog (Ekström et al. 2012). ",
#         "We confine the hypocentral depths of earthquakes in training and testing datasets to a maximum of 70km"]
#     mdFile.new_list(items = data, marked_with='-')
#     mdFile.new_line("The observed catalog from "+ str(experiment_config['start_date']) + " to " + str(experiment_config['end_date']) +":")
#     mdFile.new_line(mdFile.new_inline_image(text='Observed Catalog', path=result_figs_paths['catalog']))
#
#
#
#     mdFile.new_header(level=2, title="Competing Forecast Models")
#     mdFile.new_paragraph("TIme independent global seismicity models are provided.The earthquake forecasts are "
#                      "rate-based, i.e. forecast are provided as earthquake rates for each longitude/latitute/magnitude bins "
#                      "with bin size of 0.1. All forecasting models compute earthquake rates within unique depth bin, i.e. [0 km, 70km]"
#                      )
#
#     mdFile.new_line(text="Following forecast models are competing in this gefe")
#     model_list = ["GEAR1 (Bird et al. 2015)",
#                  "WHEEL (Bayona et al. 2021)",
#                  "TEAM (Bayona et al. 2021)",
#                  "KJSS (Kagan and Jackson (2011))",
#                  "SHIFT_GSRM2F (Bird & Kreemer (2015))"]
#     mdFile.new_list(items =model_list, marked_with='-')
#
#     mdFile.new_header(level=2, title='Quadtree Forecasts')
#     mdFile.new_paragraph(" The forecast models are provided for the classical CSEP grid. "
#                      "We want to evaluate all these forecasts for multi-resolution grids. "
#                      "We proposed Quadtree to generate multi-resolution grids. "
#                      "We provided different data-driven multi-resolution grids based "
#                      "on earthquake catalog and strain data points. "
#                      "We aggregate all the classical forecasts on Quadtree grids. "
#                      "Every forecast on a different Quadtree grid is treated as an "
#                      "independent forecast and evaluated independently to study the "
#                      "performance of forecast models for different grids.")
#
# #mdFile.new_header(level=2, title="WHEEL")
# #
# #mdFile.new_header(level=2, title="GEAR1")
# #
# #mdFile.new_header(level=2, title="KJSS")
# #
# #mdFile.new_header(level=2, title="TEAM")
# #
# #mdFile.new_header(level=2, title="SHIFT_GSRM")
#
#
#
#     mdFile.new_header(level=2, title="Evaluations")
#
#     mdFile.new_header(level=3, title='N-test')
#
#     mdFile.new_line(mdFile.new_inline_image(text='N-Test', path=result_figs_paths['N-Test']))
#     mdFile.new_line("The results of N-test from "+ str(experiment_config['start_date']) + " to " + str(experiment_config['end_date']))
#     mdFile.new_line("The models passing the N-test are marked with blue dots, while models failing the test are marked with red.")
#     #mdFile.new_line(mdFile.new_inline_image(text=image_text, path=path))
#
#     mdFile.new_header(level=3, title='CL-Test')
#
#     mdFile.new_line(mdFile.new_inline_image(text='CL-Test', path=result_figs_paths['CL-Test'])) #
#     mdFile.new_line("The results of CL-test from "+ str(experiment_config['start_date']) + " to " + str(experiment_config['end_date']))
#     mdFile.new_line("The test shows overall spatial-magnitude consistency of the forecast model with the observation. "
#                          "The models showing lower observed likelihood than the confidence interval of of "
#                          "log-likelihoods values are unable to pass the CL-test.")
#     # mdFile.new_line('  - Inline link: ' + mdFile.new_inline_link(link=link, text=text))
#
#
#     mdFile.new_header(level=3, title='M-test')
#
#     mdFile.new_line(mdFile.new_inline_image(text ='M-Test', path=result_figs_paths['M-Test']))
#     mdFile.new_line("The results of M-test from "+ str(experiment_config['start_date']) + " to " + str(experiment_config['end_date']))
#     mdFile.new_line("The test shows the consistency of magnitude aspect of the forecast with the observation. "
#                          "The models showing lower observed likelihood than the confidence interval of of "
#                          "log-likelihoods values are unable to pass the M-test.")
#
#
#
#
#
#     mdFile.new_header(level=3, title= 'S-test')
#
#     mdFile.new_line(mdFile.new_inline_image(text = 'S-Test', path = result_figs_paths['S-Test']))
#     mdFile.new_line("The results of S-test from "+ str(experiment_config['start_date']) + " to " + str(experiment_config['end_date']))
#     mdFile.new_line("The test shows the consistency of spatial aspect of the forecast with the observation. "
#                          "The models showing lower observed likelihood than the confidence interval of of "
#                          "log-likelihoods values are unable to pass the S-test.")
#
#
#     mdFile.new_header(level=3, title = 'T-test')
#
#     mdFile.new_line(mdFile.new_inline_image(text = 'T-Test', path = result_figs_paths['T-Test']))
#     mdFile.new_line("The results of comparative T-test from "+ str(experiment_config['start_date']) + " to " + str(experiment_config['end_date']))
#     mdFile.new_line("The mean information gain per earthquake as is shownc ircles, and the 95 percent confidence interval with vertical lines. "
#                          "The models with information gain higher than zero are more informative than the benchmark model, while models "
#                          "with lower information gain are less informative.")
#
#
#
#     mdFile.new_table_of_contents(table_title='Contents', depth=3)
#     return mdFile
#
# result_figs_paths = {'catalog': 'code/results/test_catalog.png',
#                 'N-Test': 'code/results/quadtree_global_experimentN-Test.png',
#                 'CL-Test': 'code/results/quadtree_global_experimentCL-Test.png',
#                 'M-Test': 'code/results/quadtree_global_experimentM-Test.png',
#                 'S-Test': 'code/results/quadtree_global_experimentS-Test.png',
#                 'T-Test': 'code/results/quadtree_global_experimentT-Test.png' }
#
# experiment_configuration = ['ABC', 'YXZ', 'CDF', 'DEF', 'GHI']
#
#
# mark_down = MdUtils(file_name='GEFE_Markdown', title='Global Earthquake Forecast Experiment (GEFE)')
#
#
#
# mark_down = generate_markdown_report(mark_down, result_figs_paths, experiment_config)
#
# mark_down.create_md_file()
