import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.stats import linregress

# Define the slope type!!! ("p" for positive, "n" for negative)
transport_type = "n"

# Define the path to the folder containing the Excel files
folder_path = r"C:\My files\Imperial-local\data\OPT data\QIDS - OPT\PEI Teflon"
output_file = r"C:\My files\Imperial-local\data\Python save path\changes-QIDS-PEI-{}.xlsx".format(transport_type)
power_density_file = r"C:\My files\Imperial-local\data\OPT data\Power density for Python display 780nm.xlsx"

# Device parameters
length = 30
width = 1000
capacitance = 5E-9  # PMMA and Teflon

# Initialize an empty DataFrame to hold all the data
all_data = pd.DataFrame()

# Function to perform natural sorting
def natural_sort_key(s):
    return [float(text) if text.replace('.', '', 1).isdigit() else text.lower() for text in re.split('(\d+\.\d+|\d+)', s)]

# Get all the .xls and .xlsx files in the specified folder and sort them naturally
files_to_process = [f for f in os.listdir(folder_path) if f.endswith(('.xls', '.xlsx'))]
files_to_process.sort(key=natural_sort_key)

print("Files to process:", files_to_process)

# Read the power density data
power_density_df = pd.read_excel(power_density_file)
power_density_dict = dict(zip(power_density_df.iloc[:, 0], power_density_df.iloc[:, 1]))

# Function to process a file and return the results as a DataFrame
def process_file(file_path, intensity):
    try:
        # Read the Excel file
        df = pd.read_excel(file_path, engine='xlrd' if file_path.endswith('.xls') else 'openpyxl')
        
        # Strip any whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Rename columns to standardize them if possible
        if 'GateV' in df.columns and 'DrainI' in df.columns:
            df['GateV(2)'] = df['GateV']
            df['DrainI(2)'] = df['DrainI']
            df['DrainI(1)'] = df['DrainI']
        elif 'GateV(1)' in df.columns and 'DrainI(1)' in df.columns:
            df['GateV(2)'] = df['GateV(1)']
            df['DrainI(2)'] = df['DrainI(1)']

        # Read DrainV column
        if 'DrainV(1)' in df.columns:
            VD = df['DrainV(1)']
        elif 'DrainV' in df.columns:
            VD = df['DrainV']
        else:
            VD = None
        VD = abs(VD)

        # Check if the required columns are present
        if 'GateV(2)' in df.columns and VD is not None:
            if 'DrainI(2)' in df.columns:
                abs_ID2 = df['DrainI(2)'].abs()
            else:
                abs_ID2 = df['DrainI'].abs() if 'DrainI' in df.columns else None
            
            abs_ID1 = df['DrainI(1)'].abs() if 'DrainI(1)' in df.columns else abs_ID2
            V_G = df['GateV(2)']
            
            # Ensure GateV(2) values are sorted and unique
            V_G, unique_indices = np.unique(V_G, return_index=True)
            abs_ID2 = abs_ID2.iloc[unique_indices] if abs_ID2 is not None else None
            abs_ID1 = abs_ID1.iloc[unique_indices]
            VD = VD.iloc[unique_indices]
            
            if abs_ID2 is not None:
                # Calculate sqrt(abs_ID2)
                sqrt_abs_ID2 = np.sqrt(abs_ID2)
                
                # Find the steepest slope using linear regression on segments for sqrt(abs_ID2)
                max_slope_ID2 = None
                window_size = 5  # Number of points to include in each linear fit
                threshold_voltage = None
                
                for i in range(len(V_G) - window_size + 1):
                    slope2, intercept2, _, _, _ = linregress(V_G[i:i + window_size], sqrt_abs_ID2[i:i + window_size])
                    
                    if transport_type == "p":
                        if max_slope_ID2 is None or slope2 < max_slope_ID2:
                            max_slope_ID2 = slope2
                            threshold_voltage = -intercept2 / slope2  # Calculate x-intercept for threshold voltage
                    else:
                        if max_slope_ID2 is None or slope2 > max_slope_ID2:
                            max_slope_ID2 = slope2
                            threshold_voltage = -intercept2 / slope2  # Calculate x-intercept for threshold voltage
                
                # Calculate the saturation mobility using the steepest slope for sqrt(abs_ID2)
                if max_slope_ID2 is not None:
                    sat_mobility = (max_slope_ID2**2) * 2 * length / (capacitance * width)
                else:
                    sat_mobility = None
            else:
                sat_mobility = None
                threshold_voltage = None
            
            # Calculate the first differential of ID1 vs VG
            dID1_dVG = np.diff(abs_ID1) / np.diff(V_G)
            
            # Find the maximum differential for linear mobility
            if dID1_dVG.size > 0:
                max_dID1_dVG = np.max(abs(dID1_dVG))
            else:
                max_dID1_dVG = None
            
            # Calculate the linear mobility using the maximum differential
            if max_dID1_dVG is not None:
                lin_mobility = length * max_dID1_dVG / (capacitance * width * VD.iloc[:-1].mean())
            else:
                lin_mobility = None
            
            # Round intensity to one decimal place
            intensity_rounded = round(intensity, 1)
            
            # Map intensity to power density
            power_density = power_density_dict.get(intensity_rounded, None)
            if intensity == 0:
                power_density = 0
            
            # Append the result to the DataFrame
            result = pd.DataFrame({
                'Power Density': [power_density],
                'Sat. Mobility': [sat_mobility],
                'Lin. Mobility': [lin_mobility],
                'Threshold Voltage': [threshold_voltage]
            })
            return result
        else:
            print(f"Required columns not found in {file_path}")
            print("Available columns:", df.columns)
            return pd.DataFrame()
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return pd.DataFrame()

# Function to process the dark file and return the results as a DataFrame
def process_dark_file(file_path):
    return process_file(file_path, intensity=0)

# Process each file and concatenate the results
for file_name in files_to_process:
    file_path = os.path.join(folder_path, file_name)
    
    if "dark" in file_name.lower():
        # Process the dark file
        result = process_dark_file(file_path)
    else:
        # Extract intensity from the file name
        intensity = float(re.findall(r'\d+\.\d+|\d+', file_name)[0])
        result = process_file(file_path, intensity)
    
    all_data = pd.concat([all_data, result], ignore_index=True)

# Save the combined data to a new Excel file
if not output_file.endswith('.xlsx'):
    output_file += '.xlsx'

if not all_data.empty:
    all_data.to_excel(output_file, index=False)
    print("Combined data saved to:", output_file)
else:
    print("No data to save.")

def plot_results(df):
    # Extract data
    power_density = df['Power Density']
    sat_mobility = df['Sat. Mobility']
    lin_mobility = df['Lin. Mobility']
    threshold_voltage = df['Threshold Voltage']
    
    # Calculate Z-scores
    z_scores_sm = zscore(sat_mobility)
    z_scores_lm = zscore(lin_mobility)
    z_scores_tv = zscore(threshold_voltage)
    
    # Define a threshold for Z-score to identify outliers
    threshold = 3
    
    # Filter out outliers
    filtered_sm = sat_mobility[np.abs(z_scores_sm) < threshold]
    filtered_power_density_sm = power_density[np.abs(z_scores_sm) < threshold]
    
    filtered_lm = lin_mobility[np.abs(z_scores_lm) < threshold]
    filtered_power_density_lm = power_density[np.abs(z_scores_lm) < threshold]
    
    filtered_tv = threshold_voltage[np.abs(z_scores_tv) < threshold]
    filtered_power_density_tv = power_density[np.abs(z_scores_tv) < threshold]
    
    # Create figure and axis objects
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))

    # Plot saturation mobility (blue) without outliers
    ax[0].semilogx(filtered_power_density_sm, filtered_sm, 'o-', color='blue', label='Saturation Mobility')
    ax[0].set_ylabel(r'$\mu_{sat}$ (cm$^2$ V$^{-1}$ s$^{-1}$)', fontsize=20)
    ax[0].grid(False)
    ax[0].set_title('Light Sensitivity Analysis', fontsize=22)
    ax[0].tick_params(axis='both', which='major', labelsize=18)
    y_min_sm = np.min(filtered_sm)
    y_max_sm = np.max(filtered_sm)
    margin_sm = (y_max_sm - y_min_sm) * 0.05
    ax[0].set_ylim(y_min_sm - margin_sm, y_max_sm + margin_sm)
    
    # Plot linear mobility (red) without outliers
    ax[1].semilogx(filtered_power_density_lm, filtered_lm, 'o-', color='red', label='Linear Mobility')
    ax[1].set_ylabel(r'$\mu_{lin}$ (cm$^2$ V$^{-1}$ s$^{-1}$)', fontsize=20)
    ax[1].grid(False)
    ax[1].set_xlabel(r'Power Density (W cm$^{-2}$)', fontsize=20)
    ax[1].tick_params(axis='both', which='major', labelsize=18)
    y_min_lm = np.min(filtered_lm)
    y_max_lm = np.max(filtered_lm)
    margin_lm = (y_max_lm - y_min_lm) * 0.05
    ax[1].set_ylim(y_min_lm - margin_lm, y_max_lm + margin_lm)
    
    # Plot threshold voltage (green) without outliers
    ax[2].semilogx(filtered_power_density_tv, filtered_tv, 'o-', color='black', label='Threshold Voltage')
    ax[2].set_ylabel(r'$V_{th}$ (V)', fontsize=20)
    ax[2].grid(False)
    ax[2].set_xlabel(r'Power Density (W cm$^{-2}$)', fontsize=20)
    ax[2].tick_params(axis='both', which='major', labelsize=18)
    y_min_tv = np.min(filtered_tv)
    y_max_tv = np.max(filtered_tv)
    margin_tv = (y_max_tv - y_min_tv) * 0.05
    ax[2].set_ylim(y_min_tv - margin_tv, y_max_tv + margin_tv)
    
    plt.tight_layout()
    plt.show()

if not all_data.empty:
    plot_results(all_data)
