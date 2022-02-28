#/usr/bin/sh
git clone https://git.gfz-potsdam.de/csep-group/global_forecasts.git

output_dir=${1:-models}
echo $output_dir
mkdir -p $output_dir
compressed_models_path=$(dirname "$0")/global_forecasts/models
# extract models
tar -xvzf $compressed_models_path/SHIFT2F_GSRM_csep.tar.gz -C $output_dir
tar -xvzf $compressed_models_path/WHEELr_csep.tar.gz -C $output_dir
tar -xvzf $compressed_models_path/TEAMr_csep.tar.gz -C $output_dir
tar -xvzf $compressed_models_path/KJSS_csep.tar.gz -C $output_dir
tar -xvzf $compressed_models_path/GEAR1_csep.tar.gz -C $output_dir
