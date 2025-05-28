import os
import logging
from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)

class MinioManager:
    def __init__(self):
        self.endpoint = os.environ.get('MINIO_ENDPOINT')
        self.access_key = os.environ.get('MINIO_ACCESS_KEY')
        self.secret_key = os.environ.get('MINIO_SECRET_KEY')
        self.bucket_name = os.environ.get('MINIO_BUCKET_NAME', 'nocobase-backups')
        self.secure = os.environ.get('MINIO_SECURE', 'False').lower() == 'true' # Default to False (HTTP) for local dev

        if not all([self.endpoint, self.access_key, self.secret_key, self.bucket_name]):
            logger.error("MinIO environment variables not fully configured. MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET_NAME are required.")
            self.client = None
            return

        try:
            self.client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure 
            )
            logger.info(f"MinIO client initialized for endpoint: {self.endpoint}, bucket: {self.bucket_name}, secure: {self.secure}")
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {e}")
            self.client = None

    def _ensure_bucket_exists(self):
        if not self.client:
            logger.error("MinIO client not initialized. Cannot ensure bucket exists.")
            return False
        try:
            found = self.client.bucket_exists(self.bucket_name)
            if not found:
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Bucket '{self.bucket_name}' created successfully.")
            else:
                logger.info(f"Bucket '{self.bucket_name}' already exists.")
            return True
        except S3Error as e:
            logger.error(f"S3Error when checking or creating bucket '{self.bucket_name}': {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error when checking or creating bucket '{self.bucket_name}': {e}")
            return False

    def upload_file(self, source_filepath: str, destination_filename: str) -> bool:
        if not self.client:
            logger.error("MinIO client not initialized. Cannot upload file.")
            return False
        
        if not os.path.exists(source_filepath):
            logger.error(f"Source file not found for MinIO upload: {source_filepath}")
            return False

        if not self._ensure_bucket_exists():
            logger.error(f"Failed to ensure MinIO bucket '{self.bucket_name}' exists. Aborting upload.")
            return False

        try:
            self.client.fput_object(
                self.bucket_name,
                destination_filename, # This is the object name in MinIO
                source_filepath
            )
            logger.info(f"Successfully uploaded '{source_filepath}' to MinIO bucket '{self.bucket_name}' as '{destination_filename}'.")
            return True
        except S3Error as e:
            logger.error(f"S3Error during file upload to MinIO ('{source_filepath}' to '{destination_filename}'): {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during file upload to MinIO ('{source_filepath}' to '{destination_filename}'): {e}")
            return False

if __name__ == '__main__':
    # Example usage (requires environment variables to be set)
    # Before running, set:
    # export MINIO_ENDPOINT='localhost:9000'
    # export MINIO_ACCESS_KEY='minioadmin'
    # export MINIO_SECRET_KEY='minioadmin'
    # export MINIO_BUCKET_NAME='mytestbucket'
    # export MINIO_SECURE='False' # or 'True' if using HTTPS

    logging.basicConfig(level=logging.INFO)
    
    # Create a dummy file for testing
    dummy_file_path = "test_upload.txt"
    with open(dummy_file_path, "w") as f:
        f.write("This is a test file for MinIO upload.")

    manager = MinioManager()
    if manager.client:
        # Example: Upload with a specific destination filename
        success = manager.upload_file(dummy_file_path, "remote_test_upload.txt")
        if success:
            print(f"File '{dummy_file_path}' uploaded successfully as 'remote_test_upload.txt'.")
        else:
            print(f"Failed to upload '{dummy_file_path}'. Check logs.")
        
        # Example: Upload with a destination filename including "folders"
        success_nested = manager.upload_file(dummy_file_path, "my_collection/archive/remote_test_upload_nested.txt")
        if success_nested:
            print(f"File '{dummy_file_path}' uploaded successfully as 'my_collection/archive/remote_test_upload_nested.txt'.")
        else:
            print(f"Failed to upload '{dummy_file_path}' to nested path. Check logs.")
    else:
        print("MinIO client could not be initialized. Check MinIO server running and env vars.")

    # Clean up dummy file
    if os.path.exists(dummy_file_path):
        os.remove(dummy_file_path)
