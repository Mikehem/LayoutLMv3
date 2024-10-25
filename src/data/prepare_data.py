import os
import json
import torch
from tqdm import tqdm
from PIL import Image
import pytesseract
from src.data.pretrain_preprocessing import prepare_pretrain_data
from src.data.funsd_preprocessing import process_funsd_dataset
import logging
import numpy as np
import sys
import traceback
import yaml

logger = logging.getLogger(__name__)

def prepare_pretraining_data(config):
    data_dir = config['data']['dir']
    images_dir = os.path.join(data_dir, 'images')
    ocr_dir = os.path.join(data_dir, 'ocr')
    processed_dir = os.path.join(data_dir, 'prepared')
    
    os.makedirs(ocr_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    processed_files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
    
    if len(image_files) != len(processed_files):
        logger.info("Preparing pretraining data...")
        
        try:
            for image_file in tqdm(image_files, desc="Processing images"):
                image_path = os.path.join(images_dir, image_file)
                ocr_path = os.path.join(ocr_dir, f"{os.path.splitext(image_file)[0]}.json")
                processed_path = os.path.join(processed_dir, f"{os.path.splitext(image_file)[0]}.pt")
                
                if not os.path.exists(ocr_path):
                    # Run OCR
                    image = Image.open(image_path)
                    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                    with open(ocr_path, 'w') as f:
                        json.dump(ocr_data, f)
                
                # Process image and OCR data
                with open(ocr_path, 'r') as f:
                    ocr_data = json.load(f)
                
                processed_data = prepare_pretrain_data(image_path, ocr_data, config)
                
                # Save processed data as a state dict
                torch.save({k: v for k, v in processed_data.items() if isinstance(v, torch.Tensor)}, 
                           processed_path)
            
            logger.info(f"Pretraining data preparation completed. {len(image_files)} files processed.")
        except Exception as e:
            logger.error("An error occurred during pretraining data preparation:")
            logger.error(traceback.format_exc())
            sys.exit(1)  # Stop execution with error code 1
    else:
        logger.info("Pretraining data already prepared.")

def prepare_funsd_data(config):
    logger.info("Preparing FUNSD data...")
    try:
        # Process training data
        train_dir = config['finetuning']['train_dir']
        train_prepared_dir = os.path.join(train_dir, 'prepared')
        os.makedirs(train_prepared_dir, exist_ok=True)
        process_funsd_dataset(train_dir, train_prepared_dir, config)
        logger.info(f"Processed FUNSD training data. Output: {train_prepared_dir}")

        # Process validation/testing data
        eval_dir = config['finetuning']['eval_dir']
        eval_prepared_dir = os.path.join(eval_dir, 'prepared')
        os.makedirs(eval_prepared_dir, exist_ok=True)
        process_funsd_dataset(eval_dir, eval_prepared_dir, config)
        logger.info(f"Processed FUNSD evaluation data. Output: {eval_prepared_dir}")

        logger.info("FUNSD data preparation completed.")
    except Exception as e:
        logger.error(f"An error occurred during FUNSD data preparation: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)  # Exit the program if there's an error

def prepare_data(config):
    try:
        prepare_pretraining_data(config)
        prepare_funsd_data(config)
    except Exception as e:
        logger.error(f"An error occurred during data preparation: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        prepare_data(config)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
