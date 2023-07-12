#!/bin/bash

python plot-error-distributions.py \
  --config-files ./configs/pyatmos__polynomial.yaml ./configs/pyatmos__pca.yaml ./configs/pyatmos__our-method.yaml \
  --output-file-name pyatmos.pdf ;

python plot-error-distributions.py \
  --config-files ./configs/goyal-2020__polynomial.yaml ./configs/goyal-2020__pca.yaml ./configs/goyal-2020__our-method.yaml \
  --output-file-name goyal-2020.pdf ;
