# floatcsep run-docker

### Docker Execution

1. Build the Docker image:

   ```bash
   docker build \

--build-arg USER_UID=$(id -u) \
--build-arg USER_GID=$(id -g) \
-t floatcsep-hermes .

   ```

2. Run the container:


   ```bash
   docker run --rm \
     -u $(id -u):$(id -g) \
     floatcsep-hermes
   ```

## Output Files

The system generates several output files in the `output/csep/` directory:

- `NGSTEP_daily_0.txt`: Main forecast output file

## File Structure

```
.
├── Dockerfile
├── docker-entrypoint.sh
├── runSTEP.sh
├── step_config.yaml
├── Update_parameters.py
├── ModifyCatalog.py
└── STEP/
    └── OpenSHA/
        ├── build/
        │   ├── lib/
        │   └── bin/
        ├── data/
        │   └── csep/
        └── output/
            └── csep/
```

## Requirements

### Java Dependencies

- commons-httpclient.jar
- commons-logging.jar
- commons-math-1.1.jar
- dom4j.jar
- jargs.jar
- jquakeml-1.0.1-2.0.1-RC1.jar
- log4j-1.2.4.jar
- step-aftershock.jar

### System Requirements

- Java Runtime Environment
- Python 3.x (for parameter updates and catalog processing)
- GMT (for visualization)

