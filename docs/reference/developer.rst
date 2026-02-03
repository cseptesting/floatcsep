Developer Notes
===============

Last updated: 03 February 2026

Creating a new release of floatCSEP
--------------------------------

These are the steps required to create a new release of floatCSEP. CI tools will automatically
bump the version on `PyPI` and `conda-forge`. Note: permissions are required to push new versions to `PyPI`.
A release is simply done when Tagging and Releasing through Github. The specific steps are:

1. Code changes
***************

1. Start a Draft Release in the Github repository once the codebase is ready. Create a tag with the new verison number and generate the change logs automatically. Keep it as draft.
2. Create a new branch and PR (or a new commit) and checkout.
3. Update `CREDITS.md <https://github.com/cseptesting/floatcsep/blob/master/CREDITS.md>`_ with the authors that changed the codebase in betweeen the releases.
4. Update `codemeta.json <https://github.com/cseptesting/floatcsep/blob/master/codemeta.json>`_
5. Issue a pull request that contains these changes.
6. Merge pull request when all changes are merged into `main` and versions are correct.
7. Publish the release
