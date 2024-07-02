import pandas as pd
import json
import argparse
import os


def load_json_to_dataframe(json_filepath):
    # Load the JSON file
    with open(json_filepath, "r") as file:
        modified_data = json.load(file)

    # Extract column names
    columns = modified_data["columns"]

    # Extract data
    data = modified_data["data"]

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)

    return df


def main(json_path):
    # Load JSON to DataFrame
    df = load_json_to_dataframe(json_path)

    # Print the first few rows of the DataFrame
    print(df.head())

    # Generate the CSV file path
    csv_path = os.path.splitext(json_path)[0] + ".csv"

    # Save DataFrame to CSV
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON to CSV")
    parser.add_argument("json_filepath", help="Path to the JSON file to be processed")
    args = parser.parse_args()

    main(args.json_filepath)
