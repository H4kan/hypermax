import pandas as pd


for i in range(0,200):
# Load the CSV file into a DataFrame
    df = pd.read_csv(f'benchmarking/atpe3_Hartmann3_{i}.csv')

    # Keep the first row and every even-indexed row (0-based index)
    filtered_df = df.iloc[::2]

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(f'benchmarking/atpe3_Hartmann3_{i}.csv', index=False)