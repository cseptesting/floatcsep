<!-- omit in toc -->
# Contributing to floatcsep

First off, thanks for taking the time to contribute! ❤️

All types of contributions are encouraged and valued. See the [Table of Contents](#table-of-contents) for different ways to help and details about how this project handles them. Please make sure to read the relevant section before making your contribution. It will make it a lot easier for us maintainers and smooth out the experience for all involved. The community looks forward to your contributions. 🎉

> And if you like the project, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project
> - Refer this project in your project's readme
> - Mention the project at local meetups and tell your friends/colleagues

<!-- omit in toc -->
## Table of Contents

- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
- [Styleguides](#styleguides)
  - [Commit Messages](#commit-messages)
- [Join The Project Team](#join-the-project-team)



## I Have a Question

> If you want to ask a question, we hope that you have alredy read the available [Documentation](https://floatcsep.readthedocs.io).

Before you ask a question, it is best to search for existing [Issues](https://github.com/cseptesting/floatcsep/issues) that might help you. In case you have found a suitable issue and still need clarification, you can write your question in this issue.

If you then still feel the need to ask a question and need clarification, we recommend the following:

- Open an [Issue](https://github.com/cseptesting/floatcsep/issues/new).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions (nodejs, npm, etc), depending on what seems relevant.
- Provide the files/artifacts, or simplified versions of them

We will then take care of the issue as soon as possible.


## I Want To Contribute

> ### Legal Notice <!-- omit in toc -->
> When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content and that the content you contribute may be provided under the project license.

### Reporting Bugs

<!-- omit in toc -->
#### Before Submitting a Bug Report

Please complete the following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version (or a tag release).
- Determine if your bug is really a bug and not an error on your side e.g. using incompatible environment components/versions (Make sure that you have read the [documentation](https://floatcsep.readthedocs.io). If you are looking for support, you might want to check [this section](#i-have-a-question)).
- To see if other users have experienced (and potentially already solved) the same issue you are having, check if there is not already a bug report existing for your bug or error in the [bug tracker](https://github.com/cseptesting/floatcsepissues?q=label%3Abug).
- Collect information about the bug:
  - Stack trace (Traceback)
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - Possibly your input and the expected output
  - Can you reliably reproduce the issue?
  
> You must never report security related issues, vulnerabilities or bugs including sensitive information to the issue tracker, or elsewhere in public. Instead, sensitive bugs must be sent by email to <pciturri@gfz-potsdam.de>.

<!-- omit in toc -->


### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for floatcsep, **including completely new features and minor improvements to existing functionality**. Following these guidelines will help maintainers and the community to understand your suggestion and find related suggestions.

#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation](https://floatcsep.readthedocs.io) carefully and find out if the functionality is already covered, maybe by an individual configuration.
- Perform a [search](https://github.com/cseptesting/floatcsep/issues) to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.
- Use a **clear and descriptive title** for the issue to identify the suggestion.
- **Describe the current behavior** and **explain which behavior you expected to see instead** and why. At this point you can also tell which alternatives do not work for you.
- **Explain why this enhancement would be useful** to most floatcsep users. You may also want to point out the other projects that solved it better and which could serve as inspiration.


### Your First Code Contribution

* Make sure you have an active GitHub account
* Fork the repo on GitHub. It will now live at `https://github.com/<YOUR_GITHUB_USERNAME>/floatcsep` ([here is some helping info](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/working-with-forks)).
* Download and install `git`. Check out the git documentaion if you aren't familiar with it.
* Please follow the [Installation](https://floatcsep.readthedocs.io) instructions for developers. Here is a summarized version.
  
      # clone your fork
      git clone https://github.com/<YOUR_GITHUB_USERNAME>/floatcsep.git
      cd floatcsep
      # prepare environment
      conda env create -n floatcsep-dev
      conda activate floatcsep-dev
      conda install -c conda-force pycsep
      # install floatcsep
      pip install -e .[dev]
      # add upstream repository
      git remote add upstream https://github.com/cseptesting/floatcsep.git

  * Note: use the command `conda deactivate` to go back to your regular environment when you are done working with floatCSEP.

  You can now do any local changes in your `floatcsep` source code, which you can then `add`, `commit` and `push` to your personal fork.

### Submitting a Pull Request

Pull requests are how we use your changes to the code! Please submit them to us! Here's how:

1. Make a new branch. For features/additions base your new branch at `main`.
2. Make sure to add tests! Only pull requests for documentation, refactoring, or plotting features do not require a test.
3. Also, documentation must accompany new feature requests.
   - Note: We would really appreciate pull requests that help us improve documentation.
4. Make sure the tests pass. Run `pytest -v tests/` in the top-level directory of the repo.
5. Push your changes to your fork and submit a pull request. Make sure to set the branch to `floatcsep:main`.
6. Wait for our review. There may be some suggested changes or improvements. Changes can be made after
   the pull request has been opening by simply adding more commits to your branch.

Pull requests can be changed after they are opened, so create a pull request as early as possible.
This allows us to provide feedback during development and to answer any questions.


Also, if you find floatCSEP to be useful, but don't want to contribute to the code we highly encourage updates to the documentation!




## Additional Resources
* [Working with Git Forks](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/working-with-forks)
* [Style Guide](http://google.github.io/styleguide/pyguide.html)
* [Docs or it doesn’t exist](https://lukeplant.me.uk/blog/posts/docs-or-it-doesnt-exist/)
* [Quickstart guide for Sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html)
* [Pep8 style guide](https://pep8.org/)
* Performance Tips:
  * [Python](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
  * [NumPy and ctypes](https://scipy-cookbook.readthedocs.io/)
  * [SciPy](https://www.scipy.org/docs.html)
  * [NumPy Book](http://csc.ucdavis.edu/~chaos/courses/nlp/Software/NumPyBook.pdf)



## Attribution
This guide is based on the **contributing-gen**. [Make your own](https://github.com/bttger/contributing-gen)!
