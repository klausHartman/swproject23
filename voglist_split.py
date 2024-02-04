import pandas as pd

# Replace 'your_file_path' with the actual path to your file
file_path = '../VOGDB/vog.members.tsv'

# Read the tab-separated file into a DataFrame
df = pd.read_csv(file_path, sep='\t')

# Define a function to split the comma-separated values into a list
def split_protein_ids(ids):
    return ids.split(',')

# Apply the function to the "ProteinIDs" column and create a new column "ProteinIDList"
df['ProteinIDList'] = df['ProteinIDs'].apply(split_protein_ids)

# Display the updated DataFrame with the new column
print(df['ProteinIDList'][5])
