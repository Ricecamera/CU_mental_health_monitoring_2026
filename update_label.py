## for loop to load csv file in data directory
import os
import pandas as pd

train_dir = 'data/train'
test_dir = 'data/test'

# list file recursively in data directory
for filename in os.listdir(train_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(train_dir, filename)
        df = pd.read_csv(file_path)

        # Check if 'status' column exists
        if 'status' in df.columns:
            mapping_value = {'Normal' : 'Normal', 'Anxiety' : 'Negative', 'Depression' : 'Very Negative', 'Suicidal' : 'Suicidal'}
            df['status'] = df['status'].map(mapping_value)
            
            # Save the updated DataFrame back to the CSV file
            df.to_csv(file_path, index=False)
            print(f"Updated status in {filename}")
        else:
            print(f"'status' column not found in {filename}")
        

# Repeat the same process for test directory
for filename in os.listdir(test_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(test_dir, filename)
        df = pd.read_csv(file_path)

        print(f"Processing {filename}...")
        print(df.head())
        
        # Check if 'status' column exists
        if 'status' in df.columns:
            mapping_value = {'Normal' : 'Normal', 'Anxiety' : 'Negative', 'Depression' : 'Very Negative', 'Suicidal' : 'Suicidal'}
            df['status'] = df['status'].map(mapping_value)
            
            # Save the updated DataFrame back to the CSV file
            df.to_csv(file_path, index=False)
            print(f"Updated status in {filename}")
        else:
            print(f"'status' column not found in {filename}")

