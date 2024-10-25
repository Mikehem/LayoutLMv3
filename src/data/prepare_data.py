import os
import json
import torch
from tqdm import tqdm
from PIL import Image
import pytesseract
from src.data.pretrain_preprocessing import prepare_pretrain_data
import logging
import numpy as np
import sys
import traceback

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
