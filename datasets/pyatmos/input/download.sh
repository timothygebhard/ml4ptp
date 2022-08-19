#!/bin/bash

# This script can be used to the PyATMOS data from the exoplanet archive.
# It is only a lightly modified version of the script provided on the website
# of the PyATMOS data set:
#
#     https://exoplanetarchive.ipac.caltech.edu/docs/fdl_landing.html
#
# Note that the full dataset is pretty large (more than 100 GB). Even with a
# fast internet connection, downloading and unpacking can take a few hours.

# Start stopwatch
printf "\nDOWNLOAD PYATMOS DATA FROM EXOPLANET ARCHIVE\n\n"
start=$(date +%s)

# Download data from exoplanet archive (ca. 40 GB)
wget -O pyatmos_summary.csv 'https://exoplanetarchive.ipac.caltech.edu/data/FDL/ATMOS/pyatmos_summary.csv'
wget -O dir_0.tar.gz 'https://exoplanetarchive.ipac.caltech.edu/data/FDL/ATMOS/tarfiles/dir_0.tar.gz'
wget -O dir_1.tar.gz 'https://exoplanetarchive.ipac.caltech.edu/data/FDL/ATMOS/tarfiles/dir_1.tar.gz'
wget -O dir_2.tar.gz 'https://exoplanetarchive.ipac.caltech.edu/data/FDL/ATMOS/tarfiles/dir_2.tar.gz'
wget -O dir_3.tar.gz 'https://exoplanetarchive.ipac.caltech.edu/data/FDL/ATMOS/tarfiles/dir_3.tar.gz'
wget -O dir_4.tar.gz 'https://exoplanetarchive.ipac.caltech.edu/data/FDL/ATMOS/tarfiles/dir_4.tar.gz'
wget -O dir_5.tar.gz 'https://exoplanetarchive.ipac.caltech.edu/data/FDL/ATMOS/tarfiles/dir_5.tar.gz'
wget -O dir_6.tar.gz 'https://exoplanetarchive.ipac.caltech.edu/data/FDL/ATMOS/tarfiles/dir_6.tar.gz'
wget -O dir_7.tar.gz 'https://exoplanetarchive.ipac.caltech.edu/data/FDL/ATMOS/tarfiles/dir_7.tar.gz'
wget -O dir_8.tar.gz 'https://exoplanetarchive.ipac.caltech.edu/data/FDL/ATMOS/tarfiles/dir_8.tar.gz'
wget -O dir_9.tar.gz 'https://exoplanetarchive.ipac.caltech.edu/data/FDL/ATMOS/tarfiles/dir_9.tar.gz'
wget -O Dir_alpha.tar.gz 'https://exoplanetarchive.ipac.caltech.edu/data/FDL/ATMOS/tarfiles/Dir_alpha.tar.gz'

# Unpack data (ca. 110 GB)
tar -xvf dir_0.tar.gz
tar -xvf dir_1.tar.gz
tar -xvf dir_2.tar.gz
tar -xvf dir_3.tar.gz
tar -xvf dir_4.tar.gz
tar -xvf dir_5.tar.gz
tar -xvf dir_6.tar.gz
tar -xvf dir_7.tar.gz
tar -xvf dir_8.tar.gz
tar -xvf dir_9.tar.gz
tar -xvf Dir_alpha.tar.gz

# Delete tar files
rm dir_0.tar.gz
rm dir_1.tar.gz
rm dir_2.tar.gz
rm dir_3.tar.gz
rm dir_4.tar.gz
rm dir_5.tar.gz
rm dir_6.tar.gz
rm dir_7.tar.gz
rm dir_8.tar.gz
rm dir_9.tar.gz
rm Dir_alpha.tar.gz

# Print total runtime
end=$(date +%s)
runtime=$((end - start))
printf "\nDone! This took %s seconds.\n\n" "$runtime"
