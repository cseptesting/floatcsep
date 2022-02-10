#/usr/bin/sh
git clone https://git.gfz-potsdam.de/csep-group/global_forecasts.git
cd global_forecasts
sh extract_models.sh
cd codes
python3 import_models.py