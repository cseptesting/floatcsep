# Install Docker image from trusted source
FROM python:3.8.13

# Setup user id and permissions.
ARG USERNAME=modeler
ARG USER_UID=1100
ARG USER_GID=$USER_UID
RUN groupadd --non-unique -g $USER_GID $USERNAME \
    && useradd -u $USER_UID -g $USER_GID -s /bin/sh -m $USERNAME

# Set up work directory in the Docker container.
## *Change {pymock} to {model_name} when used as template*
WORKDIR /usr/src/pymock/

# Copy the repository from the local machine to the Docker container.
## *Only the needed folders/files for the model build*
COPY --chown=$USER_UID:$USER_GID pymock ./pymock/
COPY --chown=$USER_UID:$USER_GID tests ./tests/
COPY --chown=$USER_UID:$USER_GID setup.cfg run.py setup.py ./

# Set up and create python virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install the pymock package.
## *Uses pip to install setup.cfg and requirements/instructions therein*
RUN pip install --no-cache-dir --upgrade pip
RUN pip install .

# Docker can now be initialized as user
USER $USERNAME

