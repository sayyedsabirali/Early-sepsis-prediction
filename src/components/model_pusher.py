import sys
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
        self.s3_estimator = S3ModelEstimator(
            bucket_name=model_pusher_config.bucket_name,
            model_path=model_pusher_config.model_registry_key
,
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            if not self.model_evaluation_artifact.is_model_accepted:
                logging.info("Model not accepted. Skipping push to S3.")
                return None

            logging.info("Pushing new Sepsis model to S3")

            self.s3_estimator.save_model(
                from_file=self.model_evaluation_artifact.trained_model_path
            )

            artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.model_registry_key
,
                message="Sepsis model successfully pushed to S3",
            )

            logging.info("Model successfully pushed to S3")
            return artifact

        except Exception as e:
            raise MyException(e, sys)
