import pydicom
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

# Define your patients and their scans
patients = {
    "Patient 1": {
        "folder": "Documents_2",
        "scans": [
            "20260320171532", "20260325113333", "20260325115437", 
            "20260325115455", "20260325121528", "20260325121538", 
            "20260325123326", "20260325123344"
        ]
    },
    "Patient 4": {
        "folder": "Archive_5",
        "scans": [
            "20260327094323", "20260327094405", "20260327101212", 
            "20260327101241", "20260327103432", "20260327103447"
        ]
    },
    "Patient 5": {
        "folder": "Archive_5",
        "scans": [
            "20260327115817", 
            "20260327115832", "20260327121419", "20260327121500", 
            "20260327123359", "20260327123411", "20260327125036", 
            "20260327125046"
        ]
    },
    "Patient 6": {
        "folder": "Archive_8",
        "scans": [
            "20260327154531", "20260327154459", "20260327160712", 
            "20260327160729", "20260327162930", "20260327163015", 
            "20260327163008", "20260327163019"
        ]
    },
    "Patient 7": {
        "folder": "Archive_7",
        "scans": [
            "20260330113058", "20260330113330", "20260330120133", 
            "20260330120145", "20260330121927", "20260330122039", 
            "20260330124329", "20260330124335"
        ]
    },
    "Patient 8": {
        "folder": "Archive_11",
        "scans": [
            "20260331103015", "20260331103025", "20260331104709", 
            "20260331104724", "20260331110142", "20260331110109", 
            "20260331111530", "20260331111511"
        ]
    }
}

# Collect all images grouped by patient
patient_images = {}
for patient_name, patient_data in patients.items():
    folder_name = patient_data["folder"]
    scan_numbers = patient_data["scans"]
    
    if not os.path.exists(folder_name):
        print(f"WARNING: Folder {folder_name} not found, skipping {patient_name}")
        continue
    
    images = []
    for scan_num in scan_numbers:
        # All files have .dcm extension
        filepath = os.path.join(folder_name, f"{scan_num}.dcm")
        
        if not os.path.exists(filepath):
            print(f"  WARNING: {filepath} not found")
            continue
        
        try:
            ds = pydicom.dcmread(filepath)
            img = ds.pixel_array
            # Handle multi-frame or color images
            if len(img.shape) > 2:
                if img.shape[2] == 3:  # RGB
                    img = np.mean(img, axis=2)  # Convert to grayscale
                else:
                    img = img[0]  # Take first frame
            images.append((scan_num, img))
        except Exception as e:
            print(f"  Error reading {filepath}: {e}")
    
    if images:
        patient_images[patient_name] = images
        print(f"{patient_name}: Loaded {len(images)} images from {folder_name}")
    else:
        print(f"{patient_name}: No readable images found")

if not patient_images:
    print("ERROR: No images loaded! Check your folder names and file paths.")
    exit()

# Calculate total subplots needed
total_plots = sum(len(imgs) for imgs in patient_images.values())
n_patients = len(patient_images)

print(f"\nCreating figure: {n_patients} patients, {total_plots} images total...")

# Create master figure
fig = plt.figure(figsize=(total_plots * 1.2, n_patients * 4))

# Create GridSpec with gaps between patient rows
gs = gridspec.GridSpec(n_patients, 1, 
                       figure=fig,
                       hspace=0.3)  # Space between patients

row = 0
for patient_name, images in patient_images.items():
    n_imgs = len(images)
    
    # Create a sub-grid for this patient's images (no space between them)
    sub_gs = gridspec.GridSpecFromSubplotSpec(
        1, n_imgs, 
        subplot_spec=gs[row, 0],
        wspace=0.0, hspace=0.0
    )
    
    for idx, (scan_num, img) in enumerate(images):
        ax = fig.add_subplot(sub_gs[0, idx])
        ax.imshow(img, cmap='gray', aspect='auto')
        ax.set_title(f"{scan_num[-8:]}", fontsize=8, pad=1)
        ax.axis('off')
        ax.margins(0)
    
    # Add patient label on the left
    fig.text(0.01, gs[row, 0].get_position(fig).y0 + 
             gs[row, 0].get_position(fig).height/2,
             patient_name, fontsize=10, fontweight='bold',
             va='center', ha='right', rotation=0)
    
    row += 1

plt.suptitle("All Patients - DICOM Scans", fontsize=14, y=0.995)
plt.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.02)

# Save as PNG
output_file = "all_patients_scans.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {output_file}")
