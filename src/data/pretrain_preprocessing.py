import torch
from PIL import Image
import pytesseract
import numpy as np
import logging
import traceback
from transformers import BertTokenizerFast

logger = logging.getLogger(__name__)

def process_image(image_path, ocr_data):
    try:
        image = Image.open(image_path)
        
        tokens = ocr_data['text']
        confidences = ocr_data['conf']
        
        left = ocr_data['left']
        top = ocr_data['top']
        width = ocr_data['width']
        height = ocr_data['height']
        
        bbox = list(zip(left, top, width, height))
        
        # Filter out low-confidence tokens and empty strings
        filtered_data = [(t, b) for t, c, b in zip(tokens, confidences, bbox) if int(c) > 60 and t.strip()]
        
        if not filtered_data:
            logger.warning(f"No valid tokens found in {image_path}")
            return [], []
        
        tokens, bbox = zip(*filtered_data)
        
        return list(tokens), list(bbox)
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return [], []

def tokenize_and_align_labels(tokenizer, tokens, bboxes, max_length):
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding='max_length',
        truncation=True,
        max_length=max_length
    )

    # Align bounding boxes
    aligned_bboxes = []
    word_ids = encoding.word_ids()
    if word_ids is not None:
        for bbox, word_idx in zip(bboxes, word_ids):
            if word_idx is None:
                aligned_bboxes.append([0, 0, 0, 0])
            else:
                aligned_bboxes.append(bbox)
    else:
        # If there are no word_ids, use a default bounding box for all tokens
        aligned_bboxes = [[0, 0, 0, 0]] * len(encoding['input_ids'])

    encoding['bbox'] = aligned_bboxes

    return encoding

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * ((bbox[0] + bbox[2]) / width)),
        int(1000 * ((bbox[1] + bbox[3]) / height)),
    ]

def prepare_pretrain_data(image_path, ocr_data, config):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    tokens, bboxes = process_image(image_path, ocr_data)
    
    encoding = tokenize_and_align_labels(
        tokenizer, 
        tokens, 
        bboxes, 
        max_length=config['model']['max_seq_length']
    )
    
    # Normalize bounding boxes
    image = Image.open(image_path)
    width, height = image.size
    normalized_bboxes = [normalize_bbox(bbox, width, height) for bbox in encoding['bbox']]
    
    # Prepare tensor data
    input_ids = torch.tensor(encoding['input_ids'])
    attention_mask = torch.tensor(encoding['attention_mask'])
    bbox = torch.tensor(normalized_bboxes)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'bbox': bbox,
        'pixel_values': image_path  # Save the image path instead of the processed tensor
    }
