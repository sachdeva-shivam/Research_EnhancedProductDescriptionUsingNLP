import boto3
import pandas as pd
from transformers import DistilBertTokenizer
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO

# AWS S3 Setup
s3 = boto3.client('s3')
bucket_name = "pdbucketprocess"
data_prefix = "archive/"  # S3 prefix where your CSV files are stored

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
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image_transform(image)
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

# Function to download CSV file from S3
def download_csv_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(BytesIO(response['Body'].read()))

# Function to upload CSV file back to S3
def upload_csv_to_s3(bucket, key, dataframe):
    csv_buffer = BytesIO()
    dataframe.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())

# Load and preprocess all datasets
all_texts = []
all_images = []

# List CSV files in the S3 bucket under the specified prefix
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=data_prefix)
if 'Contents' in response:
    for obj in response['Contents']:
        file_key = obj['Key']
        if file_key.endswith("csv"):  # Process only specific CSV files
            print(f"Processing: {file_key}")
            # Load the CSV file from S3
            dataset = download_csv_from_s3(bucket_name, file_key)
            
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
                        print(f"Skipping invalid image URL: {url}")
                else:
                    print(f"Empty image URL at index: {idx}")

            # Filter the dataset to keep only valid rows
            dataset = dataset.iloc[valid_rows]

            # Upload the updated CSV back to S3
            upload_csv_to_s3(bucket_name, file_key, dataset)

            all_images.extend(processed_images)

# Combine all preprocessed data
print(f"Total Texts: {len(all_texts)}")
print(f"Total Images: {len(all_images)}")
