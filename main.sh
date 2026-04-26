#!/bin/bash

#If not already done so, perform the ultrasound validation calculation by running Ultrasound_variation R script
#This is done to calculate the ultrasound-measured volumes and save the ground truth accordingly.


input_file="BladderVolFYP_MasterDataset_First8.csv"

tail -n +2 "$input_file" | while IFS=',' read -r patient_id folder us1 us2 bv1 us3 us4 pvr csv1 us5 us6 bv3 csv2 us7 us8 bv4 csv3 us9 us10 bv5 csv4; do
    
    echo ""
    echo "-----------------------------------------"
    echo "Processing Patient $patient_id"
    echo "Folder: $folder"
    echo "-----------------------------------------"
    
    # ✓ ADD THIS: Strip quotes from variables that might have them
    folder="${folder//\"/}"
    csv1="${csv1//\"/}"
    csv2="${csv2//\"/}"
    csv3="${csv3//\"/}"
    csv4="${csv4//\"/}"
    # Also strip from any other variables that might be quoted
    bv1="${bv1//\"/}"
    bv3="${bv3//\"/}"
    bv4="${bv4//\"/}"
    bv5="${bv5//\"/}"
    pvr="${pvr//\"/}"

    # Store bladder volumes for later validation (if needed)
    bladder_volumes=("$bv1" "$pvr" "$bv3" "$bv4" "$bv5")
    
    # Process each CSV file for this patient
    csv_files=("$csv1" "$csv2" "$csv3" "$csv4")
    
    for i in "${!csv_files[@]}"; do
        csv_name="${csv_files[$i]}"
        csv_num=$((i + 1))
        
        # Skip if CSV name is empty or NA
        if [ -z "$csv_name" ] || [ "$csv_name" = "NA" ]; then
            echo "Skipping CSV$csv_num - no data available"
            continue
        fi
        
        echo ""
        echo ">>> Processing CSV$csv_num: $csv_name"
        
        # ============================================
        # SCRIPT 1: Data Preprocessing
        # ============================================
        echo "  [1/5] Running Data Preprocessing..."
        python3 DataPreprocessing3.py "Data/SensorDataRd1/$folder/optics_data_${csv_name}_part1.csv"
        
        if [ $? -ne 0 ]; then
            echo "  ERROR: Data Preprocessing failed for $csv_name"
            continue
        fi
        echo "  ✓ Preprocessing complete"
        
        # ============================================
        # SCRIPT 2: Temporal Overlap
        # ============================================
        echo "  [2/5] Running Temporal Overlap..."
        python3 TemporalOverlap.py "Preprocessed_Data/optics_data_${csv_name}_part1.cleaned.csv"
        
        if [ $? -ne 0 ]; then
            echo "  ERROR: Temporal Overlap failed for $csv_name"
            continue
        fi
        echo "  ✓ Temporal Overlap complete"
        
        # ============================================
        # SCRIPT 3: Run Feature Extraction
        # ============================================
        echo "  [3/5] Running Next Script..."
        python3 FeatureExtraction.py "TO_Data/optics_data_${csv_name}_part1.cleaned_windows.csv"
         echo "  ✓ Feature Extraction complete"
        
        # ============================================
        # SCRIPT 4: Your fourth script
        # ============================================
        echo "  [4/5] Running Fourth Script..."
        python3 FeatureSelector.py "Feature_Extracted/optics_data_${csv_name}_part1.cleaned_windows_features.csv"
         echo "  ✓ FeatureSelector complete"
        
        # ============================================
        # SCRIPT 5: Run Feature engineering
        # ============================================
        echo "  [5/5] Running Fifth Script..."
        python3 y_features.py  "Feature_Extracted/optics_data_${csv_name}_part1.cleaned_windows_features.csv"
        echo "  ✓ Lag/windowed feature complete"
        
        echo ">>> Completed $csv_name"
        
    done
        # ============================================
        # SCRIPT 6: Model
        # ============================================
    bv6="NA" #Quick-fix for earlier error
    
    echo "  [6/6] Running Sixth Script..."
    python3 MLmodel.py  "Feature_Extracted_y/optics_data_${csv1}_part1.cleaned_windows_features.csv" \\
    "Feature_Extracted_y/optics_data_${csv2}_part1.cleaned_windows_features.csv" "Feature_Extracted_y/optics_data_${csv3}_part1.cleaned_windows_features.csv" \\
    "Feature_Extracted_y/optics_data_${csv4}_part1.cleaned_windows_features.csv" $bv1 $bv3 $bv4 $bv5 $bv6 $pvr
    echo "  ✓ Model training complete"
    
    echo ""
    echo "✓ Patient $patient_id complete"
    
    echo ""
    echo "✓ Patient $patient_id complete"
    
done

echo ""
echo "========================================="
echo "Pipeline Complete!"
echo "========================================="


#Run Feature Extraction

python3 FeatureExtracion.py TO_Data/

#Run Feature Selection
python3 FeatureSelector.ipynb

#Run the feature engineering
python3 y_features.py 



#Run Model
