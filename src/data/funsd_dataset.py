import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
import logging
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)

class FUNSDDataset(Dataset):
    def __init__(self, data_dir, max_seq_length, label_map):
        self.data_dir = os.path.join(data_dir, 'prepared')
        self.max_seq_length = max_seq_length
        self.label_map = label_map
        
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Prepared data directory not found: {self.data_dir}")
        
        self.prepared_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pt')]
        
        if not self.prepared_files:
            raise ValueError(f"No prepared files found in {self.data_dir}")
        
        logger.info(f"Found {len(self.prepared_files)} prepared files in {self.data_dir}")
        
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.prepared_files)

    def __getitem__(self, idx):
        prepared_path = os.path.join(self.data_dir, self.prepared_files[idx])
        
        try:
            # Use weights_only=True to address the FutureWarning
            data = torch.load(prepared_path, map_location='cpu', weights_only=True)
            
            # Load image path separately
            with open(prepared_path.replace('.pt', '_image_path.txt'), 'r') as f:
                image_path = f.read().strip()
            data['image_path'] = image_path
        except Exception as e:
            logger.error(f"Error loading file {prepared_path}: {str(e)}")
            raise

        # Ensure all required keys are present
        required_keys = ['input_ids', 'attention_mask', 'bbox', 'labels', 'image_path']
        for key in required_keys:
            if key not in data:
                logger.error(f"Missing key '{key}' in prepared data file: {prepared_path}")
                raise KeyError(f"Missing key '{key}' in prepared data file: {prepared_path}")

        # Load and transform image
        try:
            image = Image.open(data['image_path']).convert('RGB')
            pixel_values = self.image_transform(image)
        except Exception as e:
            logger.error(f"Error processing image {data['image_path']}: {str(e)}")
            raise

        # Ensure all tensors have the correct shape
        input_ids = self.pad_or_truncate(data['input_ids'], self.max_seq_length)
        attention_mask = self.pad_or_truncate(data['attention_mask'], self.max_seq_length)
        bbox = self.pad_or_truncate_2d(data['bbox'], self.max_seq_length, 4)
        labels = self.pad_or_truncate(data['labels'], self.max_seq_length, pad_value=-100)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'bbox': bbox,
            'pixel_values': pixel_values,
            'labels': labels
        }

    def pad_or_truncate(self, tensor, target_length, pad_value=0):
        current_length = tensor.size(0)
        if current_length < target_length:
            return torch.cat([tensor, torch.full((target_length - current_length,), pad_value, dtype=tensor.dtype)])
        else:
            return tensor[:target_length]

    def pad_or_truncate_2d(self, tensor, target_length, second_dim):
        current_length = tensor.size(0)
        if current_length < target_length:
            padding = torch.zeros(target_length - current_length, second_dim, dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=0)
        else:
            return tensor[:target_length]
