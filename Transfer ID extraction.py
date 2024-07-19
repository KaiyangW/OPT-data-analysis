import pandas as pd

# Define the input and output paths
input_path = r"C:\My files\Imperial-local\data\QIDT QIDS\PFBT 10 mins Teflon\July 4th\1-3 for plot\Transfer 1-3 p.xls"
output_path = r"C:\My files\Imperial-local\data\Python save path\QIDS-PFBT-1-3-p.xlsx"

# Read the Excel file
df = pd.read_excel(input_path)

# Extract the 'GateV(1)' column and all 'DrainI(x)' columns
columns_to_extract = ['GateV(1)']
columns_to_extract += [col for col in df.columns if col.startswith('DrainI')]

# Extract the relevant columns and make a copy to avoid SettingWithCopyWarning
extracted_df = df[columns_to_extract].copy()

# Take the absolute values of all 'DrainI(x)' columns
for col in extracted_df.columns:
    if col.startswith('DrainI'):
        extracted_df.loc[:, col] = extracted_df[col].abs()

new_column_names = {'GateV(1)': '\q(V_G)'}
new_column_names.update({col: f'\q(I_D)({col[7:-1]})' for col in extracted_df.columns if col.startswith('DrainI')})
extracted_df.rename(columns=new_column_names, inplace=True)

# Save the extracted and renamed data to a new Excel file
extracted_df.to_excel(output_path, index=False)

print(f"Data extracted and saved to {output_path}")
