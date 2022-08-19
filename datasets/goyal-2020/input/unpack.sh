#! /bin/bash

# Unpack zip files (and contents) for Goyal-2020 data set.
#
# Unfortunately, it seems impossible to fully automate the downloading
# and unpacking of the data for the Goyal-2020 data set, as the data are
# hosted on Google Drive, and Google is taking strong precautions against
# automated batch downloads: The limit for the number of files that can
# be downloaded at a time is 50, and there exists no way to download an
# entire folder as a ZIP file (like it is possible via the browser).
#
# For now, the easiest solution is to *manually* download all five folders
# from the Google Drive:
#
#     https://drive.google.com/drive/folders/1zCCe6HICuK2nLgnYJFal7W4lyunjU4JE
#
# Using the manual download through a browser (i.e., right click -> "Download")
# should give you give you five folders:
#
#     1. "Chemical Abundances-<timestamp>.zip"
#     2. "Emission Spectra-<timestamp>.zip"
#     3. "Pressure-Temperature Profiles-<timestamp>.zip"
#     4. "Transmission Spectra-<timestamp>.zip"
#     5. "Utilities-<timestamp>.zip"
#
# Place them in the same folder as this script, and rename them as follows:
#
#     1. "chemical-abundances.zip"
#     2. "emission-spectra.zip"
#     3. "pressure-temperature-profiles.zip"
#     4. "transmission-spectra.zip"
#     5. "utilities.zip"
#
# Now, running this script should automatically unpack these zip files and
# their contents (lots of tar files containing lots of gz files) to allow
# further processing using `00_make-hdf-files.py`.


# Define function for unpacking a zip folder
unpack_zip_folder() {
  unzip -oq "$1.zip"
  mv "$2" "$1"
  for TAR_FILE in "$1"/*.tar
  do
    printf "  Processing %s... " "$TAR_FILE"
    tar -xf "$TAR_FILE" -C "$1"
    FOLDER=$(basename "$TAR_FILE" .tar)
    for GZ_FILE in "$1"/"$FOLDER"/*.gz
    do
      gzip -dq "$GZ_FILE"
    done
    rm "$TAR_FILE"
    printf "Done!\n"
  done
  rm "$1.zip"
}


# Start stopwatch
printf "\nUNPACK GOYAL-2020 DATA\n\n"
start=$(date +%s)

# Unpack the chemical abundances
printf "UNPACKING CHEMICAL ABUNDANCES:\n"
unpack_zip_folder "chemical-abundances" "Chemical Abundances"
printf "\n"

# Unpack the emission spectra
printf "UNPACKING EMISSION SPECTRA:\n"
unpack_zip_folder "emission-spectra" "Emission Spectra"
printf "\n"

# Unpack the pressure temperature profiles
printf "UNPACKING PRESSURE TEMPERATURE PROFILES:\n"
unpack_zip_folder "pressure-temperature-profiles" "Pressure-Temperature Profiles"
printf "\n"

# Unpack the transmission spectra
printf "UNPACKING TRANSMISSION SPECTRA:\n"
unpack_zip_folder "transmission-spectra" "Transmission Spectra"
printf "\n"

# Unpack the utilities
printf "UNPACKING UTILITIES... "
unzip -oq "utilities.zip" && mv "Utilities" "utilities" && rm "utilities.zip"
printf "Done!\n\n"

# Print total runtime
end=$(date +%s)
runtime=$((end - start))
printf "\nDone! This took %s seconds.\n\n" "$runtime"
