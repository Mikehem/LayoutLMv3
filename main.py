import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
import click
import logging
from src.trainer import train_model
from src.finetune import finetune_model
from src.inference import run_inference

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

@click.command()
@click.option('--config', prompt='Enter the path to the config file', help='Path to the config file')
@click.option('--mode', type=click.IntRange(1, 3), prompt='Choose the mode (1: Pretraining, 2: Finetuning, 3: Inference)', help='Mode of operation')
def main(config, mode):
    """LayoutLMv3 main script for pretraining, finetuning, and inference."""
    try:
        config_data = load_config(config)
        logger.info(f"Configuration loaded from {config}")

        if mode == 1:
            logger.info("Starting pretraining...")
            train_model(config_data)
        elif mode == 2:
            logger.info("Starting finetuning...")
            finetune_model(config_data)
        elif mode == 3:
            logger.info("Starting inference...")
            run_inference(config_data)
        
        mode_names = {1: "Pretraining", 2: "Finetuning", 3: "Inference"}
        logger.info(f"{mode_names[mode]} completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.exception("Traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main()
