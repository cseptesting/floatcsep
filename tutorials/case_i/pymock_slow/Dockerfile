# Install Docker image from trusted source
FROM python:3.8.13

# Setup user id and permissions.
ARG USERNAME=modeler
ARG USER_UID=1100
ARG USER_GID=$USER_UID

# User setup
RUN groupadd --gid $USER_GID $USERNAME \
 && useradd --uid $USER_UID --gid $USER_GID --shell /bin/bash --create-home $USERNAME

# Create floatcsep IO interface
RUN mkdir -p /app/input /app/forecasts && chown -R $USERNAME:$USERNAME /app

# Create user and venv
USER $USERNAME
WORKDIR /app

# Set up Python venv
ENV VIRTUAL_ENV=/home/$USERNAME/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Create virtual environment and install python basic packages
RUN python3 -m venv $VIRTUAL_ENV && pip install --upgrade pip setuptools wheel

# Copy the repository from the local machine to the Docker container.
## *Only the needed folders/files for the model build*
COPY --chown=$USER_UID:$USER_GID pymock/ ./pymock/
COPY --chown=$USER_UID:$USER_GID setup.cfg setup.py ./

# Install the pymock package.
## *Uses pip to install setup.cfg and requirements/instructions therein*
RUN pip3 install --no-cache-dir -e .

# Docker can now be initialized as user
USER $USERNAME

ENTRYPOINT ["pymock"]

