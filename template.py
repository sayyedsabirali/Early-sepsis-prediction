import os
from pathlib import Path

PROJECT_NAME = "src"

list_of_files = [

    # core package
    f"{PROJECT_NAME}/__init__.py",

    # components
    f"{PROJECT_NAME}/components/__init__.py",
    f"{PROJECT_NAME}/components/data_ingestion.py",
    f"{PROJECT_NAME}/components/data_validation.py",
    f"{PROJECT_NAME}/components/data_transformation.py",
    f"{PROJECT_NAME}/components/model_trainer.py",
    f"{PROJECT_NAME}/components/model_evaluation.py",
    f"{PROJECT_NAME}/components/model_pusher.py",

    # pipeline
    f"{PROJECT_NAME}/pipeline/__init__.py",
    f"{PROJECT_NAME}/pipeline/training_pipeline.py",
    f"{PROJECT_NAME}/pipeline/prediction_pipeline.py",

    # configuration
    f"{PROJECT_NAME}/configuration/__init__.py",
    f"{PROJECT_NAME}/configuration/mongo_db_connection.py",
    f"{PROJECT_NAME}/configuration/aws_connection.py",

    # cloud storage
    f"{PROJECT_NAME}/cloud_storage/__init__.py",
    f"{PROJECT_NAME}/cloud_storage/aws_storage.py",

    # data access
    f"{PROJECT_NAME}/data_access/__init__.py",
    f"{PROJECT_NAME}/data_access/sepsis_data.py",

    # feature store (NEW)
    f"{PROJECT_NAME}/feature_store/__init__.py",
    f"{PROJECT_NAME}/feature_store/feature_writer.py",
    f"{PROJECT_NAME}/feature_store/feature_reader.py",

    # entities
    f"{PROJECT_NAME}/entity/__init__.py",
    f"{PROJECT_NAME}/entity/config_entity.py",
    f"{PROJECT_NAME}/entity/artifact_entity.py",
    f"{PROJECT_NAME}/entity/estimator.py",
    f"{PROJECT_NAME}/entity/s3_estimator.py",

    # monitoring (NEW)
    f"{PROJECT_NAME}/monitoring/__init__.py",
    f"{PROJECT_NAME}/monitoring/data_drift.py",
    f"{PROJECT_NAME}/monitoring/model_drift.py",

    # logging & exception
    f"{PROJECT_NAME}/logger/__init__.py",
    f"{PROJECT_NAME}/exception/__init__.py",

    # utils
    f"{PROJECT_NAME}/utils/__init__.py",
    f"{PROJECT_NAME}/utils/main_utils.py",

    # experiments 
    "experiments/.gitkeep",

    # notebooks 
    "notebooks/eda.ipynb",
    "notebooks/model_analysis.ipynb",

    # tests 
    "tests/__init__.py",
    "tests/test_data_ingestion.py",
    "tests/test_model_trainer.py",

    # config
    "config/schema.yaml",
    "config/model.yaml",

    # root files
    "app.py",
    "demo.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "pyproject.toml",
    "README.md",
    "projectworkflow.txt"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir:
        os.makedirs(filedir, exist_ok=True)

    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            pass
