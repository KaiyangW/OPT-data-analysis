import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

transport_type = "n"

# Define OPT folder paths
folder_path = r"C:\My files\Imperial-local\data\OPT data\QIDS - OPT\PEI Teflon"
output_file = r"C:\My files\Imperial-local\data\Python save path\QIDS-PEI-{}.xlsx".format(transport_type)

# Define dark current paths
dark_file_path = r"C:\My files\Imperial-local\data\OPT data\QIDS - OPT\PEI Teflon\dark\dark 2-1 n.xls"
dark_save_path = os.path.join(folder_path, "0 dark.xlsx")

# Get the power density data, choose 1100nm or 780nm
# eopt_file = r"C:\My files\Imperial-local\data\OPT data\E_opt for Python 1100nm.xlsx"
eopt_file = r"C:\My files\Imperial-local\data\OPT data\E_opt for Python 780nm.xlsx" 

# Extract Responsivity values at specific V_G values

target_vg_values_p = [-25, -40, -60, -80] # p
# target_vg_values_n = [25, 40, 60, 80] #n

# target_vg_values_p = [-5, -20, -50, -80] # p
target_vg_values_n = [5, 20, 50, 80] #n

responsivity_at_vg = {}

if transport_type == 'n':
    target_vg_values = target_vg_values_n
else:
    target_vg_values = target_vg_values_p

# Device area a: 30um * 1000um, change if it's 40um TFTs
a = 3e4

# Process the individual file to generate "0 dark.xlsx"
data_dark = pd.read_excel(dark_file_path, engine='xlrd')

# Check for the exact column names
expected_columns = ["DrainI(1)", "DrainI(2)", "GateV(1)", "DrainI"]

for col in expected_columns:
    if col not in data_dark.columns:
        print(f"Warning: Column '{col}' not found in the Excel file.")

# Extract the necessary columns and compute absolute values, if they exist
drainI_1 = data_dark["DrainI(1)"].abs() if "DrainI(1)" in data_dark.columns else data_dark["DrainI"].abs() if "DrainI" in data_dark.columns else None
drainI_2 = data_dark["DrainI(2)"].abs() if "DrainI(2)" in data_dark.columns else None
gateV_1 = data_dark["GateV(1)"] if "GateV(1)" in data_dark.columns else data_dark["GaveV"] if "GateV" in data_dark.columns else None

# Check if the required columns are available for plotting
if drainI_1 is not None and gateV_1 is not None:   
    # Define a shift factor to move the curves downwards on the log scale
    shift_factor = 1.15

    # Apply the shift factor for y-axis
    drainI_1_shifted = drainI_1 / shift_factor

    # Create a new DataFrame with the shifted data
    shifted_data = {
        "GateV(1)": gateV_1,
        "DrainI(1)": -1 * drainI_1_shifted,
    }

    if drainI_2 is not None:
        drainI_2_shifted = drainI_2 / shift_factor
        shifted_data["DrainI(2)"] = -1 * drainI_2_shifted

    # Convert the dictionary to a DataFrame for further processing
    shifted_data_df = pd.DataFrame(shifted_data)

    # Save the shifted data to a new Excel file
    shifted_data_df.to_excel(dark_save_path, index=False)

    print(f"Optimized data has been saved to: {dark_save_path}")
else:
    print("Error: Required columns are not available in the Excel file.")

# Step 2: Process the folder, now that "0 dark.xlsx" has been generated

# Initialize empty DataFrames to hold the results
all_data = pd.DataFrame({"\q(V_G)": gateV_1})
photosensitivity_data = pd.DataFrame({"\q(V_G)": gateV_1})
responsivity_data = pd.DataFrame({"\q(V_G)": gateV_1})

# Function to perform natural sorting
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

# Get all the .xls and .xlsx files in the specified folder and sort them naturally
files_to_process = [f for f in os.listdir(folder_path) if f.endswith(('.xls', '.xlsx'))]
files_to_process.sort(key=natural_sort_key)

# Read Eopt values from the provided Excel file
eopt_df = pd.read_excel(eopt_file, header=None)
Eopt_values = eopt_df.iloc[:, 0].tolist()

# Define the mapping of file names to Eopt values based on the power column
power_values = [
    '0.1', '0.2', '0.4', '0.6', '0.8', '1', '2', '4', '6', '8',
    '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'
]
power_to_Eopt = dict(zip(power_values, Eopt_values))

# Separate dark current files and illumination files
dark_files = [f for f in files_to_process if 'dark' in f.lower()]
illumination_files = [f for f in files_to_process if 'dark' not in f.lower()]

# Process dark current files first
dark_current = None
for file_name in dark_files:
    # Construct the full file path
    file_path = os.path.join(folder_path, file_name)
    
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Extract the required columns and take absolute values
        if 'DrainI(1)' in df.columns:
            df_extracted = df[['DrainI(1)']].abs()
        elif 'DrainI' in df.columns:
            df_extracted = df[['DrainI']].abs()
        else:
            continue

        dark_current = df_extracted.iloc[:, 0].astype(float)
        break
    except Exception as e:
        print(f"Failed to process {file_name}: {e}")

# Loop through all the sorted .xls and .xlsx files
for file_name in files_to_process:
    # Construct the full file path
    file_path = os.path.join(folder_path, file_name)
    
    try:
        # Read the Excel file
        if file_name.endswith('.xls'):
            df = pd.read_excel(file_path, engine='xlrd')
        else:
            df = pd.read_excel(file_path, engine='openpyxl')
        
        # Extract the required columns and take absolute values
        if 'DrainI(1)' in df.columns:
            df_extracted = df[['DrainI(1)']].abs()
            if 'DrainI(2)' in df.columns:
                df_extracted['DrainI(2)'] = df['DrainI(2)'].abs()
        elif 'DrainI' in df.columns:
            df_extracted = df[['DrainI']].abs()
        else:
            continue

        # Rename columns to include the filename for clarity
        df_extracted.columns = [f"{file_name}_{col}" for col in df_extracted.columns]
        all_data = pd.concat([all_data, df_extracted], axis=1)
        
        # Process illumination files
        if dark_current is not None and 'dark' not in file_name.lower():
            # Extract the power value from the file name
            power = re.findall(r'\d+\.\d+|\d+', file_name)
            if not power:
                continue
            power = power[0]
            
            # Extract the required columns and take absolute values
            if 'DrainI(1)' in df.columns:
                illumination_current = df[['DrainI(1)']].abs().iloc[:, 0].astype(float)
            elif 'DrainI' in df.columns:
                illumination_current = df[['DrainI']].abs().iloc[:, 0].astype(float)
            else:
                continue
                
            # Get the corresponding Eopt value for this power
            Eopt_value = power_to_Eopt.get(power)
            if Eopt_value is None:
                continue
                
            # Calculate Photosensitivity (P) and Responsivity (R)
            P = (illumination_current - dark_current) / dark_current
            R = (illumination_current - dark_current) / (Eopt_value * a)
            
            # Add the results to the respective DataFrames
            photosensitivity_data[file_name] = P.abs()
            responsivity_data[file_name] = R.abs()

            # responsivity_data[file_name] = R
            # photosensitivity_data[file_name] = P
    except Exception as e:
        print(f"Failed to process {file_name}: {e}")

# Extract responsivity at target V_G values
for vg in target_vg_values:
    if vg in gateV_1.values:
        index = gateV_1[gateV_1 == vg].index[0]
        responsivity_at_vg[vg] = responsivity_data.iloc[index, 1:].values

# Create DataFrame for Responsivity at target V_G values
responsivity_at_vg_df = pd.DataFrame(responsivity_at_vg, index=responsivity_data.columns[1:])

# Save the combined data to a new Excel file with separate sheets
if not output_file.endswith('.xlsx'):
    output_file += '.xlsx'

with pd.ExcelWriter(output_file) as writer:
    all_data.to_excel(writer, sheet_name='DrainI Data', index=False)
    if not photosensitivity_data.empty:
        photosensitivity_data.to_excel(writer, sheet_name='Photosensitivity(P)', index=False)
    if not responsivity_data.empty:
        responsivity_data.to_excel(writer, sheet_name='Responsivity(R)', index=False)
    if not responsivity_at_vg_df.empty:
        responsivity_at_vg_df.to_excel(writer, sheet_name='Responsivity at V_G', index=True)

print("Combined data saved to:", output_file)

# Plotting

# Define colormaps for each plot
cmap_drainI = plt.get_cmap("plasma")
cmap_responsivity = plt.get_cmap("viridis")
cmap_photosensitivity = plt.get_cmap("plasma")

def plot_with_gradient(data, y_label, title, subplot_index, cmap):
    if data.empty:
        return
    colors = cmap(np.linspace(1, 0, len(data.columns) - 1))  # Reverse the gradient, excluding the first column
    plt.subplot(1, 3, subplot_index)
    for color, column in zip(colors, data.columns[1:]):  # Skip the first column
        plt.plot(gateV_1, data[column], color=color)
    plt.yscale('log')
    plt.xlabel('GateV(1) (V)')
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)

# Set up the plot
plt.figure(figsize=(21, 7))

# Plot DrainI vs GateV(1)
plot_with_gradient(all_data, 'DrainI (A)', 'DrainI vs GateV(1)', 1, cmap_drainI)

# Plot Responsivity vs GateV(1)
plot_with_gradient(responsivity_data, 'Responsivity (A/W)', 'Responsivity vs GateV(1)', 2, cmap_responsivity)

# Plot Photosensitivity vs GateV(1)
plot_with_gradient(photosensitivity_data, 'Photosensitivity (P)', 'Photosensitivity vs GateV(1)', 3, cmap_photosensitivity)

# Adjust layout
plt.tight_layout()
plt.savefig("combined_plots_horizontal.png")
plt.show()