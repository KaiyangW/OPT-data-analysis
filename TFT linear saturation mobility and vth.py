import os
import pandas as pd
import re
import numpy as np
from scipy.stats import linregress

# Define the path to the folder containing the Excel files
folder_path = r"C:\My files\Imperial-local\data\QIDT QIDS\QIDT\PEI 25 July"
output_file = r"C:\My files\Imperial-local\data\Python save path\QIDT-PEI-m.xlsx"

# Device parameters
length = 30
width = 1000
capacitance = 5E-9  # PMMA and Teflon
n=7

# Initialize empty DataFrames to hold the data for 'p' and 'n' types
all_data_p = pd.DataFrame()
all_data_n = pd.DataFrame()

# Function to perform natural sorting
def natural_sort_key(s):
    return [float(text) if text.replace('.', '', 1).isdigit() else text.lower() for text in re.split('(\d+\.\d+|\d+)', s)]

# Get all the .xls and .xlsx files in the specified folder and sort them naturally
files_to_process = [f for f in os.listdir(folder_path) if f.endswith(('.xls', '.xlsx'))]
files_to_process.sort(key=natural_sort_key)

print("Files to process:", files_to_process)

# Function to process a file and return the results as a DataFrame
def process_file(file_path, slope_type):
    try:
        # Read the Excel file
        df = pd.read_excel(file_path, engine='xlrd' if file_path.endswith('.xls') else 'openpyxl')
        
        # Strip any whitespace from column names
        df.columns = df.columns.str.strip()

        drain_i_n = f'DrainI({n})'
        
        # Check if the required columns are present
        if drain_i_n in df.columns and 'GateV(2)' in df.columns and 'DrainI(1)' in df.columns and 'GateV(1)' in df.columns and 'DrainV(1)' in df.columns:
            abs_ID2 = df[drain_i_n].abs()
            abs_ID1 = df['DrainI(1)'].abs()
            V_G2 = df['GateV(2)']
            V_G1 = df['GateV(1)']
            V_D1 = df['DrainV(1)']
            
            # Ensure GateV(2) values are sorted and unique
            V_G2, unique_indices2 = np.unique(V_G2, return_index=True)
            abs_ID2 = abs_ID2.iloc[unique_indices2]
            
            # Calculate sqrt(abs_ID2)
            sqrt_abs_ID2 = np.sqrt(abs_ID2)
            
            # Find the steepest slope using linear regression on segments for sqrt(abs_ID2)
            max_slope_ID2 = None
            window_size = 5  # Number of points to include in each linear fit
            
            for i in range(len(V_G2) - window_size + 1):
                slope, intercept, r_value, p_value, std_err = linregress(V_G2[i:i + window_size], sqrt_abs_ID2[i:i + window_size])
                if slope_type == "p":
                    if max_slope_ID2 is None or slope < max_slope_ID2:
                        max_slope_ID2 = slope
                else:
                    if max_slope_ID2 is None or slope > max_slope_ID2:
                        max_slope_ID2 = slope
            
            # Calculate the saturation mobility using the steepest slope for sqrt(abs_ID2)
            if max_slope_ID2 is not None:
                sat_mobility = (max_slope_ID2**2) * 2 * length / (capacitance * width)
            else:
                sat_mobility = None
            
            # Ensure GateV(1) values are sorted and unique
            V_G1, unique_indices1 = np.unique(V_G1, return_index=True)
            abs_ID1 = abs_ID1.iloc[unique_indices1]
            sqrt_abs_ID1 = np.sqrt(abs_ID1)
            
            # Calculate the first differential of abs(ID1) vs VG
            dID1_dVG = np.diff(abs_ID1) / np.diff(V_G1)
            
            # Find the maximum differential for linear mobility
            if dID1_dVG.size > 0:
                max_dID1_dVG = np.max(np.abs(dID1_dVG))
            else:
                max_dID1_dVG = None
            
            # Calculate the linear mobility using the maximum differential
            if max_dID1_dVG is not None:
                lin_mobility = abs(length * max_dID1_dVG / (capacitance * width * V_D1.iloc[:-1].mean()))
            else:
                lin_mobility = None
            
            # Find the steepest slope using linear regression on segments for sqrt(abs(ID1)) for threshold voltage
            max_slope_ID1_sqrt = None
            threshold_voltage = None
            
            for i in range(len(V_G1) - window_size + 1):
                slope, intercept, r_value, p_value, std_err = linregress(V_G1[i:i + window_size], sqrt_abs_ID1[i:i + window_size])
                if slope_type == "p":
                    if max_slope_ID1_sqrt is None or slope < max_slope_ID1_sqrt:
                        max_slope_ID1_sqrt = slope
                        threshold_voltage = -intercept / slope
                else:
                    if max_slope_ID1_sqrt is None or slope > max_slope_ID1_sqrt:
                        max_slope_ID1_sqrt = slope
                        threshold_voltage = -intercept / slope
            
            # Append the result to the appropriate DataFrame
            result = pd.DataFrame({
                'Filename': [os.path.basename(file_path)],
                'Linear Mobility': [lin_mobility],
                'Saturation Mobility': [sat_mobility],
                'Threshold Voltage': [threshold_voltage]
            })
            return result
        else:
            print(f"Required columns not found in {file_path}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return pd.DataFrame()

# Process each file and concatenate the results
for file_name in files_to_process:
    file_path = os.path.join(folder_path, file_name)
    
    # Detect the slope type from the file name
    if file_name.count('n') >= 2:
        slope_type = "n"
    elif file_name.count('p') >= 1:
        slope_type = "p"
    else:
        continue  # Skip files that don't match either type

    result = process_file(file_path, slope_type)
    
    if slope_type == "p":
        all_data_p = pd.concat([all_data_p, result], ignore_index=True)
    else:
        all_data_n = pd.concat([all_data_n, result], ignore_index=True)

# Check if any 'p' type data exists before calculating mean and standard error
if not all_data_p.empty:
    mean_lin_mobility_p = all_data_p['Linear Mobility'].mean()
    std_err_lin_mobility_p = all_data_p['Linear Mobility'].std() / np.sqrt(len(all_data_p))
    
    mean_sat_mobility_p = all_data_p['Saturation Mobility'].mean()
    std_err_sat_mobility_p = all_data_p['Saturation Mobility'].std() / np.sqrt(len(all_data_p))
    
    mean_std_data_p = pd.DataFrame({
        'Parameter': ['Linear Mobility', 'Saturation Mobility'],
        'Mean Value (p)': [mean_lin_mobility_p, mean_sat_mobility_p],
        'Standard Error (p)': [std_err_lin_mobility_p, std_err_sat_mobility_p]
    })

# Check if any 'n' type data exists before calculating mean and standard error
if not all_data_n.empty:
    mean_lin_mobility_n = all_data_n['Linear Mobility'].mean()
    std_err_lin_mobility_n = all_data_n['Linear Mobility'].std() / np.sqrt(len(all_data_n))
    
    mean_sat_mobility_n = all_data_n['Saturation Mobility'].mean()
    std_err_sat_mobility_n = all_data_n['Saturation Mobility'].std() / np.sqrt(len(all_data_n))
    
    mean_std_data_n = pd.DataFrame({
        'Parameter': ['Linear Mobility', 'Saturation Mobility'],
        'Mean Value (n)': [mean_lin_mobility_n, mean_sat_mobility_n],
        'Standard Error (n)': [std_err_lin_mobility_n, std_err_sat_mobility_n]
    })

# Prepare final DataFrame with separate columns for 'p' and 'n' type results
final_data = pd.DataFrame()
if not all_data_p.empty:
    final_data['Filename (p)'] = all_data_p['Filename']
    final_data['μ_lin (p)'] = all_data_p['Linear Mobility']
    final_data['μ_sat (p)'] = all_data_p['Saturation Mobility']
    final_data['Vth (p)'] = all_data_p['Threshold Voltage']

if not all_data_n.empty:
    final_data['Filename (n)'] = all_data_n['Filename']
    final_data['μ_lin (n)'] = all_data_n['Linear Mobility']
    final_data['μ_sat (n)'] = all_data_n['Saturation Mobility']
    final_data['Vth (n)'] = all_data_n['Threshold Voltage']

# Save the combined data to a new Excel file
with pd.ExcelWriter(output_file) as writer:
    final_data.to_excel(writer, sheet_name='Results', index=False)
    if not all_data_p.empty:
        mean_std_data_p.to_excel(writer, sheet_name='Mean and Std Error (p)', index=False)
    if not all_data_n.empty:
        mean_std_data_n.to_excel(writer, sheet_name='Mean and Std Error (n)', index=False)

print("Combined data saved to:", output_file)
