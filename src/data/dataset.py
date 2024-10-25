import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
import logging
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)

class PretrainingDataset(Dataset):
    def __init__(self, data_dir, max_seq_length):
        self.data_dir = os.path.join(data_dir, 'prepared')
        self.max_seq_length = max_seq_length
        self.prepared_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pt')]
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
        
        # Load data with weights_only=True to address the FutureWarning
        data = torch.load(prepared_path, map_location='cpu', weights_only=True)

        # Create labels for masked language modeling
        labels = data['input_ids'].clone()
        labels = self.mask_tokens(labels)

        # Ensure all tensors have the correct shape
        input_ids = self.pad_or_truncate(data['input_ids'], self.max_seq_length)
        attention_mask = self.pad_or_truncate(data['attention_mask'], self.max_seq_length)
        bbox = self.pad_or_truncate_2d(data['bbox'], self.max_seq_length, 4)
        labels = self.pad_or_truncate(labels, self.max_seq_length, pad_value=-100)

        # Handle pixel_values
        if isinstance(data['pixel_values'], str):
            # If it's a file path, load and transform the image
            image = Image.open(data['pixel_values']).convert('RGB')
            pixel_values = self.image_transform(image)
        else:
            # If it's already a tensor, ensure it's the right shape and type
            pixel_values = data['pixel_values']
            if pixel_values.shape != (3, 224, 224):
                pixel_values = transforms.Resize((224, 224))(pixel_values)
            if pixel_values.dtype != torch.float32:
                pixel_values = pixel_values.float()
            # Normalize the tensor
            pixel_values = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(pixel_values)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'bbox': bbox,
            'pixel_values': pixel_values,
            'labels': labels
        }

    def mask_tokens(self, inputs, mlm_probability=0.15):
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        """
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)  # Add batch dimension if it's missing

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return labels.squeeze(0)  # Remove batch dimension if it was added

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
