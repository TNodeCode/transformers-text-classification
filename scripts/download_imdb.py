from datasets import load_dataset
import pandas as pd
import os

# Create data directory
os.makedirs("./data/imdb", exist_ok=True)

# Load the IMDb movie reviews dataset
dataset = load_dataset("imdb")

# Convert train, test, and unsupervised splits to CSV format and save locally
for split in dataset.keys():
    # Convert to a pandas DataFrame
    df = pd.DataFrame(dataset[split])
    
    # Save to CSV
    df.to_csv(f"./data/imdb/imdb_{split}.csv", index=False)

print("Dataset downloaded and saved as CSV files.")