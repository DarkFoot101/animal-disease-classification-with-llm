from classifier.config.configuration import ConfigurationManager
from classifier.components.prepare_base_model import PrepareBaseModel
from classifier.components.training import Training
from classifier.components.prepare_callbacks import PrepareCallbacks
from classifier import logger
from classifier.constants import *
from pathlib import Path

STAGE_NAME = "Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        try:
            logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
            config = ConfigurationManager()
            prepare_callbacks_config = config.get_prepare_callbacks_config()
            prepare_callbacks = PrepareCallbacks(config=prepare_callbacks_config)
            callback_list = prepare_callbacks.get_callbacks()
            training_config = config.get_training_config()
            training = Training(config=training_config)
            training.get_base_model()
            training.train_valid_generator()
            training.train(callback_list=callback_list)
        
            logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == "__main__":
    pipeline = ModelTrainingPipeline()
    pipeline.main()