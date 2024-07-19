import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os
import glob
import re

# Define the folder paths
data_folder_path = r"C:\My files\Imperial-local\data\OPT data\QIDS - OPT\current-time\QIDS PFBT Teflon"
save_folder_path = r"C:\My files\Imperial-local\data\Python save path"
save_file_path = os.path.join(save_folder_path, "QIDS FFT results PFBT.xlsx")

# Function to perform natural sort
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# Find all Excel files in the data folder
file_paths = glob.glob(os.path.join(data_folder_path, "*.xls"))
file_paths.sort(key=natural_sort_key)

# Initialize dictionaries to hold data for saving
data_p = {'Frequency': None}
data_n = {'Frequency': None}

# Initialize subplots for p and n
fig, (ax_p, ax_n) = plt.subplots(1, 2, figsize=(15, 6))

# Colors for different V_G values
colors = ['black', 'blue', 'purple', 'red', 'green']
color_idx_p = 0
color_idx_n = 0

# V_G values for labeling
# vg_values_p = [-5, -20, -50, -80]
# vg_values_n = [5, 20, 50, 80]

vg_values_p = [-25, -40, -60, -80]
vg_values_n = [25, 40, 60, 80]

# Flags to check if there is any p or n data
has_p_data = False
has_n_data = False

# Process each file
for file_idx, file_path in enumerate(file_paths, start=1):
    df = pd.read_excel(file_path)
    time = df['Time'].values
    drain_current = df['DrainI'].values

    # Perform Fast Fourier Transform
    N = len(time)
    T = np.mean(np.diff(time))  # Assuming time steps are uniform
    yf = fft(drain_current)
    xf = fftfreq(N, T)[:N//2]

    # Calculate the Power Spectral Density
    psd = (2.0/N) * np.abs(yf[0:N//2])**2

    # Convert to Noise Spectral Density
    nsd = np.sqrt(psd)

    nsd = nsd / 5

    # Store the results in a DataFrame for saving
    vg_value = os.path.basename(file_path).split()[2]  # Adjust the index based on the file naming convention
    if ' p' in file_path:
        has_p_data = True
        column_name = f'Noise spectral density {color_idx_p + 1}'
        if color_idx_p < len(vg_values_p):
            label = f'$V_G$ = {vg_values_p[color_idx_p]}'
            ax_p.loglog(xf, nsd, label=label, color=colors[color_idx_p % len(colors)])
            color_idx_p += 1
        if data_p['Frequency'] is None:
            data_p['Frequency'] = xf
        data_p[column_name] = nsd
    elif ' n' in file_path:
        has_n_data = True
        column_name = f'Noise spectral density {color_idx_n + 1}'
        if color_idx_n < len(vg_values_n):
            label = f'$V_G$ = {vg_values_n[color_idx_n]}'
            ax_n.loglog(xf, nsd, label=label, color=colors[color_idx_n % len(colors)])
            color_idx_n += 1
        if data_n['Frequency'] is None:
            data_n['Frequency'] = xf
        data_n[column_name] = nsd

# Customize the plots if there is data
if has_p_data:
    ax_p.set_title(r'$V_D = -10 V$', fontsize=26)
    ax_p.set_xlabel('Frequency (Hz)', fontsize=24)
    ax_p.set_ylabel('Noise spectral density (A Hz$^{-1/2}$)', fontsize=24)
    ax_p.grid(False)  # Remove grid lines
    ax_p.tick_params(axis='both', which='both', direction='in', length=6, width=2, colors='black', grid_color='black', grid_alpha=0.5)
    ax_p.tick_params(axis='both', labelsize=20)
    ax_p.minorticks_off()  # Turn off minor ticks
    ax_p.legend(fontsize=22)

if has_n_data:
    ax_n.set_title(r'$V_D = 10 V$', fontsize=26)
    ax_n.set_xlabel('Frequency (Hz)', fontsize=24)
    ax_n.set_ylabel('Noise spectral density (A Hz$^{-1/2}$)', fontsize=24)
    ax_n.grid(False)  # Remove grid lines
    ax_n.tick_params(axis='both', which='both', direction='in', length=6, width=2, colors='black', grid_color='black', grid_alpha=0.5)
    ax_n.tick_params(axis='both', labelsize=20)
    ax_n.minorticks_off()  # Turn off minor ticks
    ax_n.legend(fontsize=22)

plt.tight_layout()
plt.show()

# Save data_p and data_n in separate sheets of an Excel file if there is data
with pd.ExcelWriter(save_file_path) as writer:
    if has_p_data:
        data_p_df = pd.DataFrame(data_p)
        data_p_df.to_excel(writer, sheet_name='p Data', index=False)
    if has_n_data:
        data_n_df = pd.DataFrame(data_n)
        data_n_df.to_excel(writer, sheet_name='n Data', index=False)
