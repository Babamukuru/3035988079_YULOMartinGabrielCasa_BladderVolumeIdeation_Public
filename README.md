# 3035988079_YULOMartinGabrielCasa_BladderVolumeIdeation_Public
Public version of the Github repository for the final year project of 3035988079 YULO Martin Gabriel Casa

Usage: The entire pipeline is built that users only need to run main.sh. It will sequentially run each script when needed on the appropriate data file. The pipeline automatically handles creation of data subfolders, proper output file-naming and creation, and saves any produced figures as png files.

Requirements: 
Code requires up-to-date python and R. List of necessary packages for each script are also listed at the top of the respective scripts, and are additionally listed below. For the code to also function properly, it requires a master metadata file that maps each raw sensor data with the appropriate before and after ultrasound-measured bladder volume based on their timestamp and file name. The file in question would also contain patient metadata and maps each patient with their respective raw sensor data csv files, raw bladder volume images, and timestamp and data collection period (E.g., if this was the first, second, third, or fourth wear and data measurement of the sensor).

Workflow of main.sh:

