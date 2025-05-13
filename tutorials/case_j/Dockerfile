FROM python:3.11.12-slim-bullseye

# System setup
ARG USERNAME=modeler
ARG USER_UID=1100
ARG USER_GID=1100

RUN apt-get update && apt-get install -y --no-install-recommends \
    git
# User setup
RUN groupadd --gid $USER_GID $USERNAME \
 && useradd --uid $USER_UID --gid $USER_GID --shell /bin/bash --create-home $USERNAME

# Create user and venv
USER $USERNAME
WORKDIR /app

# Set up Python venv
ENV VIRTUAL_ENV=/home/$USERNAME/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONUNBUFFERED=1

RUN python3 -m venv $VIRTUAL_ENV && pip install --upgrade pip setuptools wheel && pip install floatcsep seismostats

# Copy and install STEP
COPY --chown=$USERNAME:$USERNAME config.yml pre_process.py model_results.csv seismicityobservation.xml ./
COPY --chown=$USERNAME:$USERNAME models/ models/

ENTRYPOINT ["python3 pre_process.py", "floatcsep run config.yml"]