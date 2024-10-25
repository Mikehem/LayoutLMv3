import os
import sys
import torch
import yaml
import click
import logging
import traceback
from PIL import Image
import numpy as np
from transformers import BertTokenizerFast

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.layoutlmv3 import LayoutLMv3ForPreTraining
from src.data.pretrain_preprocessing import normalize_bbox
from src.data.dataset import PretrainingDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default values
DEFAULT_CONFIG_PATH = "/home/michaeld/workdir/LayoutLMv3/config/config.yaml"
DEFAULT_IMAGE_PATH = "/home/michaeld/workdir/LayoutLMv3/data/samples/images/21095963-Amzn:Root=1-65bb003d-779b40f42c68a1290f31d455-page_1.jpeg"
DEFAULT_TEXT = "Shradha"
DEFAULT_BBOX = "674,130,616,109"

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_bbox(bbox_string):
    return [int(coord) for coord in bbox_string.split(',')]

def preprocess_input(image_path, text, bbox, config):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    normalized_bbox = normalize_bbox(bbox, width, height)

    # Tokenize the text
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    encoding = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=config['model']['max_seq_length'])

    # Create a dummy dataset item
    dummy_item = {
        'input_ids': encoding['input_ids'][0],
        'attention_mask': encoding['attention_mask'][0],
        'bbox': torch.tensor([normalized_bbox] * config['model']['max_seq_length']),
        'pixel_values': image_path,
        'labels': encoding['input_ids'][0].clone()  # Use input_ids as dummy labels
    }

    # Use PretrainingDataset to process the item
    dataset = PretrainingDataset(config['data']['dir'], config['model']['max_seq_length'])
    processed_item = dataset[0]  # This will trigger the __getitem__ method

    # Replace the pixel_values with our actual image
    processed_item['pixel_values'] = dataset.image_transform(image).unsqueeze(0)

    # Ensure all tensors have a batch dimension
    for key in processed_item:
        if isinstance(processed_item[key], torch.Tensor):
            if processed_item[key].dim() == 1:
                processed_item[key] = processed_item[key].unsqueeze(0)
            elif processed_item[key].dim() == 2 and key == 'bbox':
                processed_item[key] = processed_item[key].unsqueeze(0)

    return processed_item

def test_model_inference(config_path, image_path, text, bbox):
    try:
        # Load configuration
        config = load_config(config_path)

        # Load the pretrained model
        model = LayoutLMv3ForPreTraining(config['model'])
        model_path = config['pretraining']['model_path']
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
            logger.info(f"Loaded pretrained model from {model_path}")
        else:
            logger.warning(f"Pretrained model not found at {model_path}. Using initialized model.")

        model.eval()

        # Preprocess the input
        inputs = preprocess_input(image_path, text, bbox, config)

        # Log input shapes
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"Input '{key}' shape: {value.shape}")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Log some output information
        if hasattr(outputs, 'prediction_scores'):
            logger.info(f"Model prediction_scores shape: {outputs.prediction_scores.shape}")
        elif hasattr(outputs, 'logits'):
            logger.info(f"Model logits shape: {outputs.logits.shape}")
        else:
            logger.warning("Model output doesn't have 'prediction_scores' or 'logits' attribute")

        if hasattr(outputs, 'loss'):
            logger.info(f"Model loss: {outputs.loss.item()}")
        else:
            logger.warning("Model output doesn't have 'loss' attribute")

        return True
    except Exception as e:
        logger.error(f"Error during model inference: {str(e)}")
        logger.error(traceback.format_exc())
        return False

@click.command()
@click.option('--config', default=DEFAULT_CONFIG_PATH, prompt='Enter the path to the config file', help='Path to the config file')
@click.option('--image', default=DEFAULT_IMAGE_PATH, prompt='Enter the path to the sample image file', help='Path to the sample image file')
@click.option('--text', default=DEFAULT_TEXT, prompt='Enter a sample text for testing', help='Sample text for testing')
@click.option('--bbox', default=DEFAULT_BBOX, prompt='Enter bounding box coordinates (x1,y1,x2,y2)', help='Bounding box coordinates as x1,y1,x2,y2')
def main(config, image, text, bbox):
    """Test LayoutLMv3 pretrained model inference"""
    logger.info("Starting pretrained model inference test...")
    
    try:
        bbox = parse_bbox(bbox)
        success = test_model_inference(config, image, text, bbox)
        
        if success:
            logger.info("Test completed successfully. The pretrained model can be loaded and used for inference.")
        else:
            logger.error("Test failed. Please check the error messages above.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
