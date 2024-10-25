import torch
from PIL import Image
import pytesseract
import numpy as np
import logging
import traceback

logger = logging.getLogger(__name__)

def process_image(image_path):
    try:
        image = Image.open(image_path)
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        tokens = ocr_data['text']
        confidences = ocr_data['conf']
        
        # Ensure bbox components are lists
        left = ocr_data['left']
        top = ocr_data['top']
        width = ocr_data['width']
        height = ocr_data['height']
        
        # Check if any of the bbox components is an int (single value)
        if isinstance(left, int):
            left = [left]
        if isinstance(top, int):
            top = [top]
        if isinstance(width, int):
            width = [width]
        if isinstance(height, int):
            height = [height]
        
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

def tokenize_and_align_labels(tokenizer, tokens, bboxes, labels, max_length):
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding='max_length',
        truncation=True,
        max_length=max_length
    )

    aligned_labels = []
    for i, label in enumerate(labels):
        word_ids = encoding.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)

    # Align bounding boxes
    aligned_bboxes = []
    for bbox, word_ids in zip(bboxes, encoding.word_ids()):
        aligned_bbox = []
        for word_idx in word_ids:
            if word_idx is None:
                aligned_bbox.append([0, 0, 0, 0])
            else:
                aligned_bbox.append(bbox[word_idx])
        aligned_bboxes.append(aligned_bbox)

    encoding['labels'] = aligned_labels
    encoding['bbox'] = aligned_bboxes

    return encoding

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling.
    """
    if isinstance(inputs, int):
        logger.warning(f"Received an integer input in mask_tokens: {inputs}")
        inputs = torch.tensor([inputs])
    elif not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs)

    labels = inputs.clone()
    
    # Ensure inputs is 2D
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
        inputs = inputs.unsqueeze(0)

    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    # Replace masked input tokens with tokenizer.mask_token_id
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # If the tensors were originally 1D, return them as 1D
    if labels.shape[0] == 1:
        labels = labels.squeeze(0)
        inputs = inputs.squeeze(0)

    return labels

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * ((bbox[0] + bbox[2]) / width)),
        int(1000 * ((bbox[1] + bbox[3]) / height)),
    ]
