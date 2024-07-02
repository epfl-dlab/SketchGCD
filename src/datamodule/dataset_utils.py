from datasets import Dataset
import pandas as pd


def create_synthetic_dataset(
    sequence_length: int = 20000, num_data_points: int = 1000
) -> Dataset:
    """
    Create a synthetic dataset with sequences of repeating digits from 0 to 9.

    Parameters:
    - N (int): The length of each sequence.
    - M (int): The number of data points.

    Returns:
    - A Hugging Face dataset object containing the synthetic data.

    With the default parameters, the size of the dataset is 1000*40000*10 = 400 MB.
    """
    # Generate data points
    sequences = []
    labels = []
    for i in range(num_data_points):
        digit = "z"
        sequence = digit * sequence_length  # Repeat the current digit N times
        sequences.append(sequence)
        labels.append(digit)

    # Create a DataFrame
    df = pd.DataFrame({"sequence": sequences, "label": labels})
    # Convert DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    return dataset


if __name__ == "__main__":
    # Example usage: Create a dataset with each sequence of length 100 and 1000 data points
    sequence_length = 100  # Length of each sequence
    num_data_points = 1000  # Number of data points
    dataset = create_synthetic_dataset(sequence_length, num_data_points)

    # Show some examples from the dataset
    print(dataset[:10])
