import torch
from PIL import Image
import json
import os
import logging
import traceback
import sys
from transformers import BertTokenizerFast
from tqdm import tqdm

logger = logging.getLogger(__name__)

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

def process_funsd_file(file_path, image_dir):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        words = []
        labels = []
        bboxes = []
        
        for item in data['form']:
            for word in item['words']:
                if word['text'].strip():  # Only process non-empty words
                    words.append(word['text'])
                    labels.append(item['label'])
                    bboxes.append(word['box'])

        # Get the image file path
        image_file = os.path.basename(file_path).replace('.json', '.png')
        image_path = os.path.join(image_dir, image_file)

        return words, labels, bboxes, image_path
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)  # Exit the program if there's an error

def tokenize_and_align_labels(tokenizer, words, labels, bboxes, max_length, label_map):
    try:
        tokenized_inputs = tokenizer(
            words,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            is_split_into_words=True,
            return_offsets_mapping=True
        )

        aligned_labels = []
        aligned_bboxes = []
        
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                aligned_labels.append(-100)
                aligned_bboxes.append([0, 0, 0, 0])
            elif word_idx != previous_word_idx:
                aligned_labels.append(label_map[labels[word_idx]])
                aligned_bboxes.append(bboxes[word_idx])
            else:
                aligned_labels.append(-100)
                aligned_bboxes.append(bboxes[word_idx])
            previous_word_idx = word_idx

        tokenized_inputs["labels"] = aligned_labels
        tokenized_inputs["bbox"] = aligned_bboxes

        return tokenized_inputs
    except Exception as e:
        logger.error(f"Error in tokenize_and_align_labels: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)  # Exit the program if there's an error

def prepare_funsd_data(file_path, image_dir, config):
    try:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        
        words, labels, bboxes, image_path = process_funsd_file(file_path, image_dir)
        
        # Normalize bounding boxes
        image = Image.open(image_path)
        width, height = image.size
        normalized_bboxes = [normalize_bbox(bbox, width, height) for bbox in bboxes]
        
        # Create label map
        label_map = {label: i for i, label in enumerate(set(labels))}
        
        encoding = tokenize_and_align_labels(
            tokenizer, 
            words, 
            labels, 
            normalized_bboxes, 
            max_length=config['model']['max_seq_length'],
            label_map=label_map
        )
        
        # Prepare tensor data
        input_ids = torch.tensor(encoding['input_ids'])
        attention_mask = torch.tensor(encoding['attention_mask'])
        bbox = torch.tensor(encoding['bbox'])
        labels = torch.tensor(encoding['labels'])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'bbox': bbox,
            'labels': labels,
            'image_path': image_path
        }
    except Exception as e:
        logger.error(f"Error in prepare_funsd_data for file {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)  # Exit the program if there's an error

def process_funsd_dataset(data_dir, output_dir, config):
    try:
        os.makedirs(output_dir, exist_ok=True)
        annotation_dir = os.path.join(data_dir, 'annotations')
        image_dir = os.path.join(data_dir, 'images')
        
        annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.json')]
        
        for filename in tqdm(annotation_files, desc="Processing FUNSD files"):
            file_path = os.path.join(annotation_dir, filename)
            processed_data = prepare_funsd_data(file_path, image_dir, config)
            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.pt")
            
            # Save only the tensor data
            torch.save({k: v for k, v in processed_data.items() if isinstance(v, torch.Tensor)}, output_file)
            
            # Save the image path separately
            with open(output_file.replace('.pt', '_image_path.txt'), 'w') as f:
                f.write(processed_data['image_path'])
    except Exception as e:
        logger.error(f"Error in process_funsd_dataset: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)  # Exit the program if there's an error

if __name__ == "__main__":
    # This can be used for testing the preprocessing
    import yaml
    
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        data_dir = 'path/to/funsd/dataset'
        output_dir = 'path/to/output/directory'
        
        process_funsd_dataset(data_dir, output_dir, config)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
