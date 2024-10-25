import os
import sys
import torch
import yaml
import argparse
from PIL import Image
import numpy as np
from transformers import BertTokenizerFast
import logging
import traceback

# Update the import paths
from src.model.layoutlmv3 import LayoutLMv3ForTokenClassification
from src.data.pretrain_preprocessing import normalize_bbox
from src.data.funsd_dataset import FUNSDDataset

logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_model(config):
    model = LayoutLMv3ForTokenClassification(config['model'])
    model.load_state_dict(torch.load(config['inference']['model_path']))
    model.eval()
    return model

def prepare_input(config):
    # Load and preprocess image
    image_path = config['inference']['image_path']
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    pixel_values = torch.tensor(np.array(image)).permute(2, 0, 1).float().unsqueeze(0)

    # Tokenize text
    text = config['inference']['text']
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    encoding = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=config['model']['max_seq_length'])
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # Prepare bbox
    bbox = config['inference']['bbox']
    width, height = image.size
    bbox = [normalize_bbox(b, width, height) for b in bbox]
    bbox = torch.tensor([bbox])

    return input_ids, attention_mask, bbox, pixel_values

def perform_inference(model, input_ids, attention_mask, bbox, pixel_values):
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values
        )
    return outputs

def run_inference(config):
    try:
        model = load_model(config)
        dataset = FUNSDDataset(config['inference']['data_dir'], config['model']['max_seq_length'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        for i in range(len(dataset)):
            sample = dataset[i]
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
            bbox = sample['bbox'].unsqueeze(0).to(device)
            pixel_values = sample['pixel_values'].unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox, pixel_values=pixel_values)

            predicted_labels = outputs.logits.argmax(-1)
            logger.info(f"Predicted labels for sample {i}: {predicted_labels}")

        logger.info("Inference completed.")
    except Exception as e:
        logger.error(f"An error occurred during inference: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

def main(config_path):
    config = load_config(config_path)
    print("Configuration loaded successfully.")

    model = load_model(config)
    print("Model loaded successfully.")

    input_ids, attention_mask, bbox, pixel_values = prepare_input(config)
    outputs = perform_inference(model, input_ids, attention_mask, bbox, pixel_values)

    print("Inference completed successfully.")
    print(f"Output shape: {outputs.prediction_scores.shape}")

    # Add any post-processing or analysis of the outputs here

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="LayoutLMv3 Inference")
        parser.add_argument("--config", type=str, required=True, help="Path to the config file")
        args = parser.parse_args()
        main(args.config)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
