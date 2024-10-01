# Dockerfile Template

# ___________________________________________________________
# 1- Install a Docker base image from official/trusted source
# ___________________________________________________________
FROM {image}:{version}

 ## Examples:
 ### Ubuntu (https://hub.docker.com/_/ubuntu)
 # FROM ubuntu:20.04

 ### Python (https://hub.docker.com/_/python)
 # FROM python:3.10.0

 ## R base (https://hub.docker.com/_/r-base/)
 # FROM r-base:4.2.0

# ____________________________________________________
# 2- Set up group and user ids inside the Docker image
# ____________________________________________________

# Set default values
ARG USERNAME={model_name}
ARG USER_UID=1100
ARG USER_GID=1100

# Create group and user
RUN groupadd --non-unique -g $USER_GID $USERNAME \
    && useradd -u $USER_UID -g $USER_GID -s /bin/sh -m $USERNAME

# ____________________
# 3- Set up filesystem
# ____________________

# Model's path in the docker image
ARG MODEL_PATH=/usr/{model_name}/
WORKDIR $MODEL_PATH

# Copy the files from the host machine (the repo) to the Docker image.
COPY --chown=$USER_UID:$USER_GID {relpaths_host} $MODEL_PATH

## Ideally, the folder/files to be copied should be explicited
# COPY --chown=$USER_UID:$USER_GID source tutorials tests docs parameters setup $MODEL_PATH

# ______________________________
# 4- Build the model source code
# ______________________________

## Provide a set of instructions to build the model inside the docker image, and run the unit tests
RUN setup && run_tests

## See tutorials in
#### Python (https://git.gfz-potsdam.de/csep-group/rise_italy_experiment/models/mockup_py)
#### R (https://git.gfz-potsdam.de/csep-group/rise_italy_experiment/models/mockup_R)

# ______________________________
# 4- Create the entry point
# ______________________________

## Docker now will be initialized as $USERNAME
USER $USERNAME

## Set the command to run the model. This should be a file found on the repo, or a binary created when building the model
ENTRYPOINT ["run"]
