import os
import io
import pyzipper
from google.cloud import storage

def process_gcp_zip():
    # 1. Setup Environment & Credentials
    # Ensure GOOGLE_APPLICATION_CREDENTIALS points to your JSON key path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("CREDENTIALS_PATH")
    password = os.getenv("PASSWORD").encode()
    
    bucket_name = "vidhi_core"
    source_blob_name = "zip_test/data.zip"  # Replace with your actual filename
    destination_prefix = "unzipped_test/"

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # 2. Download the Zip File into memory
    print(f"Downloading {source_blob_name}...")
    blob = bucket.blob(source_blob_name)
    zip_data = io.BytesIO(blob.download_as_bytes())

    # 3. Decrypt and Extract
    print("Decrypting and uploading contents...")
    with pyzipper.AESZipFile(zip_data) as zf:
        zf.setpassword(password)
        
        for file_info in zf.infolist():
            with zf.open(file_info) as extracted_file:
                # 4. Upload directly back to GCP
                target_path = f"{destination_prefix}{file_info.filename}"
                new_blob = bucket.blob(target_path)
                
                print(f"  Uploading: {target_path}")
                new_blob.upload_from_file(extracted_file)

    print("Process complete!")

if __name__ == "__main__":
    process_gcp_zip()