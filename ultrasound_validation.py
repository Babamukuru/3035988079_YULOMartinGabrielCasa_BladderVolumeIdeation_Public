import pandas as pd
import numpy as np
import os


class UltrasoundDataEntry:
    """
    Ultrasound Bladder Measurement Data Handler
    
    This class manages the collection and storage of ultrasound bladder measurements.
    It provides two input methods:
    1. Manual console entry for single measurements or manual entry of patient data
    2. CSV file import for batch measurements from a prewritten excel/csv file.
    
    All measurements use the ellipsoid formula for bladder volume calculation:
    Volume (ml) = 4/3 * π * (length/2) * (width/2) * (depth/2) * 1000
    """
  
    def __init__(self):
              """
        Initialize an empty list to store ultrasound measurement entries.
        
        Each entry will be a dictionary containing:
        - Patient identification
        - Timestamp of measurement
        - Bladder dimensions (length, width, depth in cm)
        - Calculated bladder volume (ml)
        - Additional clinical context (voiding data, patient notes, etc)
        """
        self.entries = []
    
    def manual_entry(self):
        """Console-based entry for single ultrasound measurements."""
        """
        Prompts the user for:
        - Patient ID
        - Measurement timestamp
        - Bladder dimensions (length, width, depth)
        - Voided volume (optional)
        - Clinical context
        - Notes (optional)
        
        Returns the created entry dictionary.
        """
        print("=== Ultrasound Measurement Entry ===")
        patient_id = input("Patient ID: ").strip()
        
        # Timestamp handling - critical for matching!
        # Expected format: YYYY-MM-DD HH:MM (e.g., 2024-01-15 10:30)
        timestamp_str = input("Measurement time (YYYY-MM-DD HH:MM): ").strip()
        measurement_time = pd.to_datetime(timestamp_str)
        
        # Bladder dimensions, convert to float
        length = float(input("Length (cm): "))
        width = float(input("Width (cm): "))
        depth = float(input("Depth (cm): "))
        
        # Voiding context
        voided_volume = input("Voided volume (ml, press Enter if none): ")
        voided_volume = float(voided_volume) if voided_volume else None

       # Get clinical context or other optional notes
        context = input("Context (pre_void/post_void/other): ").strip()
        
        notes = input("Notes (optional): ").strip()
        
        # Validation
        self._validate_measurements(length, width, depth)
        
         entry = {
            'patient_id': patient_id,          # Patient identifier
            'timestamp': measurement_time,     # When measurement was taken
            'length_cm': length,               # Bladder length in cm
            'width_cm': width,                 # Bladder width in cm
            'depth_cm': depth,                 # Bladder depth in cm
            'voided_volume_ml': voided_volume, # Optional voided volume
            'context': context,                # Clinical context
            'notes': notes,                    # Optional notes
            'calculated_volume_ml': calculated_volume,  # Calculated bladder volume
            'source': 'manual_entry'           # Track data source
        }
        
        self.entries.append(entry)
        print(f"✓ Added entry: {entry['calculated_volume_ml']:.1f} ml")
        return entry

def auto_entry(self, csv_filepath):
        """
        Import ultrasound measurements from a CSV file.
        
        Expected CSV format (case-sensitive column names):
        - patient_id: Patient identifier
        - measurement_time: Timestamp (YYYY-MM-DD HH:MM)
        - length_cm: Bladder length in cm
        - width_cm: Bladder width in cm
        - depth_cm: Bladder depth in cm
        - voided_volume_ml: Optional voided volume in ml
        - context: Clinical context (pre_void/post_void/other)
        - notes: Optional notes
        
        Parameters:
        csv_filepath (str): Path to the CSV file containing measurements
        
        Returns:
        list: All imported entries as dictionaries
        """
        try:
            # Read the entire CSV file into a pandas DataFrame
            # pandas automatically handles comma-separated values
            df = pd.read_csv(csv_filepath)
            
            # Inform user about successful file loading
            print(f"📁 Loaded {len(df)} entries from {csv_filepath}")
            
            # Process each row in the CSV file
            for idx, row in df.iterrows():
                # Convert the current row to a properly formatted entry
                entry = self._process_csv_row(row)
                
                # Add the entry to the collection
                self.entries.append(entry)
                
                # Provide progress feedback
                print(f"  ✓ Added entry {idx+1}: {entry['calculated_volume_ml']:.1f} ml")
            
            # Summary of import operation
            print(f"✅ Successfully added {len(df)} entries from CSV")
            
            return self.entries
            
        except FileNotFoundError:
            # Handle case where the specified CSV file doesn't exist
            print(f"❌ File not found: {csv_filepath}")
            return []
    
    def _process_csv_row(self, row):
        """
        Convert a single CSV row into a standardized ultrasound entry.
        
        This internal method extracts and formats data from a CSV row.
        It assumes the CSV has the exact column names specified in auto_entry.
        
        Parameters:
        row (pd.Series): A single row from the pandas DataFrame
        
        Returns:
        dict: Formatted ultrasound entry
        """
        # Extract and clean patient ID
        patient_id = str(row['patient_id']).strip()
        
        # Extract and parse timestamp
        # Convert string to datetime for consistency with manual entries
        timestamp_str = str(row['measurement_time']).strip()
        measurement_time = pd.to_datetime(timestamp_str)
        
        # Extract bladder dimensions
        # Convert to float for mathematical operations
        length = float(row['length_cm'])
        width = float(row['width_cm'])
        depth = float(row['depth_cm'])
        
        # Validate that dimensions are positive numbers
        self._validate_measurements(length, width, depth)
        
        # Handle optional voided volume field
        # Check if column exists and has a value
        voided_volume = None
        if 'voided_volume_ml' in row and pd.notna(row['voided_volume_ml']):
            voided_volume = float(row['voided_volume_ml'])
        
        # Extract clinical context with default value
        context = 'unknown'
        if 'context' in row and pd.notna(row['context']):
            context = str(row['context']).strip().lower()
        
        # Extract optional notes
        notes = ''
        if 'notes' in row and pd.notna(row['notes']):
            notes = str(row['notes']).strip()
        
        # Calculate bladder volume using ellipsoid formula
        calculated_volume = self._calculate_volume(length, width, depth)
        
        # Create standardized entry dictionary
        entry = {
            'patient_id': patient_id,
            'timestamp': measurement_time,
            'length_cm': length,
            'width_cm': width,
            'depth_cm': depth,
            'voided_volume_ml': voided_volume,
            'context': context,
            'notes': notes,
            'calculated_volume_ml': calculated_volume,
            'source': 'csv_import'  # Track that this came from CSV
        }
        
        return entry
    
    
    def _validate_measurements(self, length, width, depth):
        """Basic sanity checks - accept any positive number"""
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}cm")
        if width <= 0:
            raise ValueError(f"Width must be positive, got {width}cm")
        if depth <= 0:
            raise ValueError(f"Depth must be positive, got {depth}cm")
    
    def _calculate_volume(self, length, width, depth):
        """Ellipsoid formula: V = 4/3 * π * (L/2) * (W/2) * (D/2)"""  #Following ellipsoid formula to calculate a bladder volume
      #Will consult Urologist on this matter on best way/formula to attain the ground truth estimation.
        return (4/3) * np.pi * (length/2) * (width/2) * (depth/2) * 1000  # ml
    
    def save_to_csv(self, filename='ultrasound_measurements.csv'):
        """Append to growing CSV file"""
        df = pd.DataFrame(self.entries)
        
        # Append if file exists, create new otherwise
        if os.path.exists(filename):
            existing = pd.read_csv(filename)
            df = pd.concat([existing, df], ignore_index=True)
        
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} entries to {filename}")

    def get_entries(self):
        """
        Get all current entries as a pandas DataFrame.
        
        Returns:
        pd.DataFrame: All ultrasound entries in tabular format
        """
        # Convert list of dictionaries to DataFrame
        # Each dictionary becomes a row, keys become column names
        return pd.DataFrame(self.entries)
    
    def clear_entries(self):
        """
        Clear all stored entries from memory.
        
        Useful for starting fresh without creating a new instance.
        """
        self.entries = []
        print("Cleared all entries from memory")
