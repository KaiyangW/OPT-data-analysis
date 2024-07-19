import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define a choice variable (user can set this to 'p' or 'n')
choice = 'n'  # Change this to 'n' if you want to read "n Data"

# Define file paths and sheet names
file_path = r"C:\My files\Imperial-local\data\Python save path\QIDS\QIDS-PEI-{}.xlsx".format(choice)
fft_file_path = r"C:\My files\Imperial-local\data\Python save path\QIDS\QIDS FFT results PEI.xlsx"
power_density_file_path = r"C:\My files\Imperial-local\data\OPT data\Power density for Python display 780nm.xlsx"
output_file = r"C:\My files\Imperial-local\data\Python save path\QIDS-PEI-Detectivity-{}.xlsx".format(choice)
sheet_name = "Responsivity at V_G"

# Determine which sheet to read based on the choice
if choice == 'p':
    fft_sheet_name = "p Data"
elif choice == 'n':
    fft_sheet_name = "n Data"
else:
    raise ValueError("Invalid choice!")

# Read the Responsivity data
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Extract columns 2 to 5 (assuming columns are named)
columns_to_use = df.columns[1:5]
data = df[columns_to_use].copy()  # Use a copy to ensure original data is not altered

# Read the FFT results to get in_delf data
fft_df = pd.read_excel(fft_file_path, sheet_name=fft_sheet_name)

# Process the last 1000 rows and find the mean of the smallest n values in each column
last_1000_rows = fft_df.tail(500)
in_delf_data = last_1000_rows.apply(lambda x: np.mean(x.nsmallest(20)), axis=0).values[1:5]

# Define constants
A = 3e-8  # m^2

# Perform calculations for each column
d_star = []
for i, col in enumerate(columns_to_use):
    R_C = data[col]
    D_star = (np.sqrt(A) * R_C) / in_delf_data[i]
    d_star.append(D_star)

# Combine the results into a DataFrame
d_star_df = pd.DataFrame(d_star).T
d_star_df.columns = columns_to_use

# Read the power density data
power_density_df = pd.read_excel(power_density_file_path)

# Extract the second column
power_density = power_density_df.iloc[:, 1].values  # Use column B (second column)

# Remove the first element if it's zero
if power_density[0] == 0:
    power_density = power_density[1:]

# Ensure the lengths match by trimming the D* DataFrame if necessary
if len(power_density) < len(d_star_df):
    d_star_df_trimmed = d_star_df.iloc[:len(power_density)]
else:
    d_star_df_trimmed = d_star_df.copy()

# Add the power density as the first column in the DataFrame
d_star_df_trimmed.insert(0, 'Power Density', power_density[:len(d_star_df_trimmed)])

# Save the result to a new Excel file
d_star_df_trimmed.to_excel(output_file, sheet_name="D_star", index=False)

# Plot the results using log-log scale
plt.figure()
for col in columns_to_use:
    plt.scatter(power_density[:len(d_star_df_trimmed)], d_star_df_trimmed[col], label=f'VG = {col}')

plt.xlabel('Power Density (W cm$^{-2}$)')
plt.ylabel('D* (Jones)')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.title('Specific Detectivity (D*) vs Power Density')
plt.grid(True, which="major", ls="--")
plt.show()

print("Detectivity results saved to:", output_file)
