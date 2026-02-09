import sys
import time
import os
from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import S3ModelEstimator


class ModelPusher:
    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.warning_s3_estimator = S3ModelEstimator(
            bucket_name=model_pusher_config.bucket_name,
            model_path=model_pusher_config.warning_model_registry_key
        )
        self.confirmation_s3_estimator = S3ModelEstimator(
            bucket_name=model_pusher_config.bucket_name,
            model_path=model_pusher_config.confirmation_model_registry_key
        )
        self.s3_service = SimpleStorageService()

    def _upload_with_retry(self, local_path: str, s3_key: str, max_retries: int = 3):
        """Upload file with retry logic"""
        for attempt in range(max_retries):
            try:
                logging.info(f"Upload attempt {attempt + 1}/{max_retries} for {s3_key}")
                
                # Check file size
                file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
                
                if file_size_mb > 100:  # Large file
                    logging.info(f"Large file detected ({file_size_mb:.1f} MB). Using simple upload.")
                    # Use simple upload instead of multipart for reliability
                    with open(local_path, 'rb') as f:
                        self.s3_service.s3_client.put_object(
                            Bucket=self.model_pusher_config.bucket_name,
                            Key=s3_key,
                            Body=f
                        )
                else:
                    # Small file - use normal upload
                    self.s3_service.upload_file(
                        from_filename=local_path,
                        to_filename=s3_key,
                        bucket_name=self.model_pusher_config.bucket_name,
                        remove=False  # Don't remove local file
                    )
                
                logging.info(f"Successfully uploaded {s3_key}")
                return True
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logging.warning(f"Upload attempt {attempt + 1} failed: {e}")
                time.sleep(5 * (attempt + 1))  # Exponential backoff
        
        return False

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            if not (self.model_evaluation_artifact.is_warning_model_accepted and 
                   self.model_evaluation_artifact.is_confirmation_model_accepted):
                logging.info("One or both models not accepted. Skipping push to S3.")
                return None

            logging.info("Pushing dual sepsis models to S3")

            # Model paths
            warning_model_path = "artifacts/model_trainer/trained_model/warning_model.pkl"
            confirmation_model_path = "artifacts/model_trainer/trained_model/confirmation_model.pkl"
            
            # Check if models exist
            if not os.path.exists(warning_model_path):
                raise FileNotFoundError(f"Warning model not found: {warning_model_path}")
            if not os.path.exists(confirmation_model_path):
                raise FileNotFoundError(f"Confirmation model not found: {confirmation_model_path}")
            
            # Upload Warning Model (Small - 1.2 MB)
            logging.info(" Uploading Warning Model (1.2 MB)...")
            warning_success = self._upload_with_retry(
                warning_model_path,
                self.model_pusher_config.warning_model_registry_key
            )
            
            if not warning_success:
                logging.error("Failed to upload warning model")
                return None
            
            # Upload Confirmation Model (Large - 850 MB)
            logging.info(" Uploading Confirmation Model (850 MB - this may take time)...")
            
            # Check file size first
            conf_size_mb = os.path.getsize(confirmation_model_path) / (1024 * 1024)
            
            if conf_size_mb > 500:
                logging.warning(f"Confirmation model is very large ({conf_size_mb:.1f} MB).")
                choice = input("Do you want to skip uploading confirmation model? (y/n): ")
                if choice.lower() == 'y':
                    logging.warning("Skipping confirmation model upload due to size.")
                    artifact = ModelPusherArtifact(
                        bucket_name=self.model_pusher_config.bucket_name,
                        warning_s3_model_path=self.model_pusher_config.warning_model_registry_key,
                        confirmation_s3_model_path=None,  # Not uploaded
                        message="Warning model uploaded. Confirmation model skipped due to size.",
                    )
                    return artifact
            
            confirmation_success = self._upload_with_retry(
                confirmation_model_path,
                self.model_pusher_config.confirmation_model_registry_key,
                max_retries=2  # Less retries for large file
            )
            
            if confirmation_success:
                artifact = ModelPusherArtifact(
                    bucket_name=self.model_pusher_config.bucket_name,
                    warning_s3_model_path=self.model_pusher_config.warning_model_registry_key,
                    confirmation_s3_model_path=self.model_pusher_config.confirmation_model_registry_key,
                    message="Dual sepsis models successfully pushed to S3",
                )
                logging.info("Both models successfully pushed to S3")
            else:
                artifact = ModelPusherArtifact(
                    bucket_name=self.model_pusher_config.bucket_name,
                    warning_s3_model_path=self.model_pusher_config.warning_model_registry_key,
                    confirmation_s3_model_path=None,
                    message="Warning model uploaded. Confirmation model upload failed.",
                )
                logging.warning(" Only warning model uploaded. Confirmation model upload failed.")
            
            return artifact

        except Exception as e:
            logging.error(f"Model pusher failed: {e}")
            # Don't raise exception, just return None
            return None