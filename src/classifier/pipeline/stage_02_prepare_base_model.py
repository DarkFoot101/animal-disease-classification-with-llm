from classifier.config.configuration import ConfigurationManager
from classifier.components.prepare_base_model import PrepareBaseModel
from classifier import logger
from classifier.constants import *
from pathlib import Path


STAGE_NAME = "Prepare Base Model Stage"

class PrepareBaseModelPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
            config = ConfigurationManager()
            prepare_base_model_config = config.get_prepare_base_model_config()
            prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
            prepare_base_model.get_base_model()
            prepare_base_model.update_base_model()
        
            logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == "__main__":
    pipeline = PrepareBaseModelPipeline()
    pipeline.main()