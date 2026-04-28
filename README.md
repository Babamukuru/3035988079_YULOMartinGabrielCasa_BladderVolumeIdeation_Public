# 3035988079_YULOMartinGabrielCasa_BladderVolumeIdeation_Public
Public version of the Github repository for the final year project of 3035988079 YULO Martin Gabriel Casa

Usage: The entire pipeline is built that users only need to run main.sh. It will sequentially run each script when needed on the appropriate data file. The pipeline automatically handles creation of data subfolders, proper output file-naming and creation, and saves any produced figures as png files.

Requirements: 
Code requires up-to-date python and R. List of necessary packages for each script are also listed at the top of the respective scripts, and are additionally listed below. For the code to also function properly, it requires a master metadata file that maps each raw sensor data with the appropriate before and after ultrasound-measured bladder volume based on their timestamp and file name. The file in question would also contain patient metadata and maps each patient with their respective raw sensor data csv files, raw bladder volume images, and timestamp and data collection period (E.g., if this was the first, second, third, or fourth wear and data measurement of the sensor).

Workflow of main.sh:

DataPreprocessing.py


TemporalOverlap.py



FeatureExtraction.py



FeatureSelector.py


y_features.py


MLmodel.py


ScikitModel.py






OTHER FILES not incorporated into main,sh:

fyp_csv_viewer.Rmd:
Viewing csv files manually via excel or textedit was too time and resource intensive and quite inefficient, so i wrote a simple R-script to help me fiddle with csv files and inspect their file sizes, row or column counts, colnames, etc. 

This is merely for troubleshooting and assistance, and is not required for the main pipeline.

Ultrasound_validation2.Rmd:
This script takes in input of Depth, Height and Width of bladder volume images, manually measured and filled in a spreadsheet. It then calculates the bladder volume based on formula D*H*W*0.7, and appends this information to the resultant csv file. Additionally, it can filter out images labelled "0" as NA or "1" as poor quality, though doing so reduces the number of viable data as files would not have respective ground truth.
