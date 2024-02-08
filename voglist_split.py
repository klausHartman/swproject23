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
print(df['ProteinIDList'])

# checking, if the length of the newly generated List in 'ProteinIDList' matches the value of column 'ProteinCount' in the original dataframe:
for idx, row in df.iterrows():
    print("column `ProteinCount`: {} -> len of `ProteinIDList`: {}".format(df.iloc[idx]["ProteinCount"], len(df.iloc[idx]["ProteinIDList"])))
# print(len(df.iloc[0]["ProteinIDList"]))
