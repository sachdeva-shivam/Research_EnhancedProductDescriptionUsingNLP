import os
import pandas as pd
from transformers import DistilBertTokenizer
from PIL import Image
from torchvision import transforms
import requests


# Set the path to your data directory
data_dir = "/home/ubuntu/data/"

# Initialize tokenizer and image transform
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to download and preprocess images
def process_image(url):
    try:
        response = requests.get(url, stream=True)
        image = Image.open(response.raw).convert("RGB")
        return image_transform(image)
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

# Load and preprocess all datasets
all_texts = []
all_images = []

for file_name in os.listdir(data_dir):
    if file_name.endswith("csv"):  # Process only CSV files
        print(f"Processing: {file_name}")
        # Load the CSV file
        file_path = os.path.join(data_dir, file_name)
        dataset = pd.read_csv(file_path)
        
        # Preprocess text data
        text_data = dataset["name"].fillna("No description").tolist()
        if text_data:  # Ensure text_data is not empty
            tokenized_texts = tokenizer(
                text_data, 
                truncation=True, 
                padding=True, 
                max_length=128, 
                return_tensors="pt"
            )
            all_texts.append(tokenized_texts)
        
        # Preprocess image data
        image_urls = dataset["image"].fillna("").tolist()
        processed_images = []
        valid_rows = []

        for idx, url in enumerate(image_urls):
            if url:
                img = process_image(url)
                if img is not None:
                    processed_images.append(img)
                    valid_rows.append(idx)
                else:
                    # Replace the URL with a working one or remove the row
                    print(f"Skipping image URL: {url}")
                    if valid_rows:
                        url = image_urls[valid_rows[-1]]  # Use the last valid URL
                        dataset.at[idx, 'image'] = url
                        img = process_image(url)
                        if img is not None:
                            processed_images.append(img)
                            valid_rows.append(idx)
                        else:
                            print(f"Removing row with invalid image URL: {url}")
                    else:
                        print(f"Removing row with invalid image URL: {url}")
            else:
                # Handle the case where the URL is empty
                print(f"Empty image URL at index: {idx}")

        # Filter the dataset to keep only valid rows
        dataset = dataset.iloc[valid_rows]

        # Update the CSV file with valid rows only
        dataset.to_csv(file_path, index=False)

        all_images.extend(processed_images)

# Combine all preprocessed data
print(f"Total Texts: {len(all_texts)}")
print(f"Total Images: {len(all_images)}")