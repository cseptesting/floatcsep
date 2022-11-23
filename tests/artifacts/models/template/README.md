# Model Template

This repository contains the outline of a model's structure to participate in the Earthquake Forecasting Experiment for Italy.

## Directory and File Structure
The should contain at the minimum, but not limited to, the following structure:
```
model_name
|   Dockerfile
|   LICENSE
|   parameters
|   README.md
|   requirements
|   run
|   run_tests
|   setup
|_ /examples
    |_ /case1
    |_ /case2
    ...
|_ /forecasts
|_ /input
|_ /source
    |   main
    |   libs
|_ /tests
    |_ /test1
    |_ /test2
    ...
```

## Source code
All the source code, i.e. the code related to generate a forecast, should be contained within a directory (named `source` in this template repo), which ideally should be homonymous to the model. It is expected to separate the core routine (main file) that outlines the model's algorithm, from the set of libraries and utilities (e.g. catalog readers, writers, geospatial functions , etc.). Note that the main routines will be called from outside the `source` folder.

## Build

An explicit set of instructions to build the model (without the use of Docker) should be provided at the top of the `README.md` file. We encourage to encapsulate the required commands into a `setup.sh` shell file. Here, libraries can be installed from different repositories (e.g. `apt`), additional setup files could be called (e.g. `python setup.py`.



# Editing this README

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.


## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.


## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

