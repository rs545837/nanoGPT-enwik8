import os
import requests
import zipfile
import numpy as np
import pickle

# Step 1: Download enwik8 dataset
def download_enwik8(output_file_path):
    enwik8_url = "http://mattmahoney.net/dc/enwik8.zip"
    if not os.path.exists(output_file_path):
        print("Downloading enwik8 dataset...")
        with open(output_file_path, "wb") as f:
            f.write(requests.get(enwik8_url).content)
        print("Download complete.")
    else:
        print("Dataset already downloaded.")

# Step 2: Extract enwik8 dataset
def extract_enwik8(zip_file, output_file):
    if not os.path.exists(output_file):
        print("Extracting enwik8 dataset...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(output_file))
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")

# Step 3: Load and split the dataset
def load_and_split_enwik8(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()

    n = len(data)
    print(f"Dataset length: {n:,} characters")

    num_test_chars = 5000000
    train_data = data[: -2 * num_test_chars]
    valid_data = data[-2 * num_test_chars: -num_test_chars]
    test_data = data[-num_test_chars:]

    return train_data, valid_data, test_data

# Step 4: Create the vocab (character-level)
def create_vocab(train_data, valid_data, test_data):
    combined_data = train_data + valid_data + test_data  # Combine all splits
    chars = sorted(list(set(combined_data)))  # Get unique characters
    vocab_size = len(chars)

    # Create mappings from characters to integers and vice versa
    stoi = {ch: i for i, ch in enumerate(chars)}  # String to Integer (Character to ID)
    itos = {i: ch for i, ch in enumerate(chars)}  # Integer to String (ID to Character)

    print(f"Vocabulary size: {vocab_size} unique characters")

    return stoi, itos, vocab_size

# Step 5: Encode the data using character-level mapping
def encode_data(data, stoi):
    return [stoi[ch] for ch in data]

# Step 6: Save encoded data to binary files
def save_to_bin(encoded_data, output_path):
    encoded_array = np.array(encoded_data, dtype=np.uint16)  # Use uint16 to save memory
    encoded_array.tofile(output_path)

# Step 7: Save the meta.pkl file (stores the vocab and mappings)
def save_meta_file(stoi, itos, vocab_size, output_dir):
    meta = {
        'stoi': stoi,  # Character to Integer mapping
        'itos': itos,  # Integer to Character mapping
        'vocab_size': vocab_size
    }
    meta_file = os.path.join(output_dir, 'meta.pkl')
    with open(meta_file, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Meta file saved to {meta_file}")

if __name__ == "__main__":
    # Base directory where everything will be stored
    base_dir = os.path.join(os.path.dirname(__file__), 'enwik8_char')
    os.makedirs(base_dir, exist_ok=True)

    # Paths
    zip_file_path = os.path.join(base_dir, "enwik8.zip")
    extracted_file_path = os.path.join(base_dir, "enwik8")

    # Step 1: Download the dataset to data directory
    download_enwik8(zip_file_path)

    # Step 2: Extract the dataset into the data directory
    extract_enwik8(zip_file_path, extracted_file_path)

    # Step 3: Load and split the dataset
    train_data, valid_data, test_data = load_and_split_enwik8(extracted_file_path)

    # Step 4: Create the vocab from combined train, valid, and test data
    stoi, itos, vocab_size = create_vocab(train_data, valid_data, test_data)

    # Step 5: Encode the data using the combined vocab
    train_ids = encode_data(train_data, stoi)
    valid_ids = encode_data(valid_data, stoi)
    test_ids = encode_data(test_data, stoi)

    # Step 6: Save the encoded data to binary files inside the data directory
    train_bin_path = os.path.join(base_dir, 'train.bin')
    valid_bin_path = os.path.join(base_dir, 'val.bin')
    test_bin_path = os.path.join(base_dir, 'test.bin')

    save_to_bin(train_ids, train_bin_path)
    save_to_bin(valid_ids, valid_bin_path)
    save_to_bin(test_ids, test_bin_path)

    # Step 7: Save the meta.pkl file
    save_meta_file(stoi, itos, vocab_size, base_dir)

    print(f"Train data has {len(train_ids):,} tokens.")
    print(f"Validation data has {len(valid_ids):,} tokens.")
    print(f"Test data has {len(test_ids):,} tokens.")
    print(f"Data saved to {train_bin_path}, {valid_bin_path}, and {test_bin_path}")
