import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.pipeline.training_pipeline import TrainPipeline
pipe = TrainPipeline()

pipe.run_pipeline()
    