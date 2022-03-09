#!/bin/zsh

printf "\n"
printf "Downloading dataset... "
wget -q https://polybox.ethz.ch/index.php/s/NJDNvggyYIRWi9L/download -O dataset.zip;
printf "Done! \n"

printf "Unpacking dataset... "
unzip -q dataset.zip ;
printf "Done! \n"

printf "Moving files... "
mv PT_Seager_Database/readme.txt readme.txt ;
mv PT_Seager_Database/seager_timmy_log10_pressures.txt pressure_grid.txt ;
mv PT_Seager_Database/seager_timmy_parameters.txt parameters.txt ;
mv PT_Seager_Database/seager_timmy_profiles.txt temperatures.txt ;
printf "Done! \n"

printf "Deleting *.zip file... "
rm dataset.zip ;
rm -r PT_Seager_Database ;
printf "Done! \n"
printf "\n"
