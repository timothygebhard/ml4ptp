#!/bin/bash

# -----------------------------------------------------------------------------
# PyAtmos (run 1)
# -----------------------------------------------------------------------------

python plot-color-coded-z.py \
  --dataset "pyatmos" \
  --key "flux_H2O" \
  --title "H\$_\mathregular{2}\$O flux (\$\mathregular{pmol}\,/\,\mathregular{s}\,/\,\mathregular{cm}^2$)" \
  --run-dir "$ML4PTP_EXPERIMENTS_DIR/pyatmos/default/latent-size-2/runs/run_1" \
  --scaling-factor 6.02214076e11 ;  # Avogadro's constant * 1e-12, this is to convert from molecules/s/cm^2 to picomoles/s/m^2

python plot-color-coded-z.py \
  --dataset "pyatmos" \
  --key "O3" \
  --title "Mean concentration of O\$_\mathregular{3}\$ (ppm)" \
  --run-dir "$ML4PTP_EXPERIMENTS_DIR/pyatmos/default/latent-size-2/runs/run_1" \
  --scaling-factor 1e-6 ;  # convert to ppm

python plot-color-coded-z.py \
  --dataset "pyatmos" \
  --key "temperature_kelvin" \
  --title "Surface temperature (K)" \
  --run-dir "$ML4PTP_EXPERIMENTS_DIR/pyatmos/default/latent-size-2/runs/run_1"

# -----------------------------------------------------------------------------
# Goyal-2020 (run 0)
# -----------------------------------------------------------------------------

python plot-color-coded-z.py \
  --dataset "goyal-2020" \
  --key "/chemical_abundances/CH4" \
  --title "Mean log\$_\mathregular{10}\$-concentration of CH\$_\mathregular{4}\$" \
  --run-dir "$ML4PTP_EXPERIMENTS_DIR/goyal-2020/default/latent-size-2/runs/run_0" \
  --use-log

python plot-color-coded-z.py \
  --dataset "goyal-2020" \
  --key "/chemical_abundances/CO2" \
  --title "Mean log\$_\mathregular{10}\$-concentration of CO\$_\mathregular{2}\$" \
  --run-dir "$ML4PTP_EXPERIMENTS_DIR/goyal-2020/default/latent-size-2/runs/run_0" \
  --use-log

python plot-color-coded-z.py \
  --dataset "goyal-2020" \
  --key "/chemical_abundances/TiO" \
  --title "Mean log\$_\mathregular{10}\$-concentration of TiO" \
  --run-dir "$ML4PTP_EXPERIMENTS_DIR/goyal-2020/default/latent-size-2/runs/run_0" \
  --use-log
