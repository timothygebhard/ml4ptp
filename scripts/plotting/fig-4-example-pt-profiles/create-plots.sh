#!/bin/bash

# This script will run `plot-pt-profile.py` with the right arguments to create
# all the plots in Fig. 4 (and the corresponding figure in the appendix)

# PyAtmos
python plot-pt-profile.py --dataset "pyatmos" --idx 0 12156 24312 --sort-by "mse" --run-dir "$ML4PTP_EXPERIMENTS_DIR/pyatmos/default/latent-size-1/runs/run_0" ;
python plot-pt-profile.py --dataset "pyatmos" --idx 0 12156 24312 --sort-by "mse" --run-dir "$ML4PTP_EXPERIMENTS_DIR/pyatmos/default/latent-size-2/runs/run_0" ;
python plot-pt-profile.py --dataset "pyatmos" --idx 0 12156 24312 --sort-by "mse" --run-dir "$ML4PTP_EXPERIMENTS_DIR/pyatmos/default/latent-size-3/runs/run_0" ;
python plot-pt-profile.py --dataset "pyatmos" --idx 0 12156 24312 --sort-by "mse" --run-dir "$ML4PTP_EXPERIMENTS_DIR/pyatmos/default/latent-size-4/runs/run_0" ;

# Goyal-2020
python plot-pt-profile.py --dataset "goyal-2020" --idx 0 645 1289 --sort-by "mse" --run-dir "$ML4PTP_EXPERIMENTS_DIR/goyal-2020/default/latent-size-1/runs/run_0" ;
python plot-pt-profile.py --dataset "goyal-2020" --idx 0 645 1289 --sort-by "mse" --run-dir "$ML4PTP_EXPERIMENTS_DIR/goyal-2020/default/latent-size-2/runs/run_0" ;
python plot-pt-profile.py --dataset "goyal-2020" --idx 0 645 1289 --sort-by "mse" --run-dir "$ML4PTP_EXPERIMENTS_DIR/goyal-2020/default/latent-size-3/runs/run_0" ;
python plot-pt-profile.py --dataset "goyal-2020" --idx 0 645 1289 --sort-by "mse" --run-dir "$ML4PTP_EXPERIMENTS_DIR/goyal-2020/default/latent-size-4/runs/run_0" ;
