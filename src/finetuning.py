import logging
import os
import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.data.funsd_dataset import FUNSDDataset
from src.model.layoutlmv3 import LayoutLMv3, LayoutLMv3ForTokenClassification
from src.utils.helpers import set_seed, get_optimizer, get_scheduler
import traceback
from src.data.prepare_data import prepare_funsd_data
from sklearn.metrics import classification_report
import yaml
import json
import numpy as np
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FineTuner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        set_seed(config['seed'])
        logger.info(f"Random seed set to {config['seed']}")

        self.setup_data()
        self.setup_model()
        self.setup_optimization()

        # Create directory for checkpoints
        os.makedirs(self.config['finetuning']['checkpoint_dir'], exist_ok=True)
        
        # Create directory for the final model
        model_dir = os.path.dirname(self.config['finetuning']['model_path'])
        os.makedirs(model_dir, exist_ok=True)

    def setup_data(self):
        logger.info("Setting up data...")
        
        if not self.config['finetuning'].get('skip_processing', False):
            prepare_funsd_data(self.config)
        else:
            logger.info("Skipping data preparation as skip_processing is set to True")
        
        # Create a default label map if not provided in config
        if 'label_map' not in self.config['finetuning']:
            logger.warning("Label map not found in config. Creating a default label map.")
            unique_labels = set()
            for split in ['train', 'eval']:
                data_dir = self.config['finetuning'][f'{split}_dir']
                for filename in os.listdir(os.path.join(data_dir, 'annotations')):
                    with open(os.path.join(data_dir, 'annotations', filename), 'r') as f:
                        data = json.load(f)
                        unique_labels.update(item['label'] for item in data['form'])
            self.config['finetuning']['label_map'] = {label: i for i, label in enumerate(sorted(unique_labels))}
            logger.info(f"Created label map: {self.config['finetuning']['label_map']}")
        
        try:
            logger.info(f"Initializing train dataset from {self.config['finetuning']['train_dir']}")
            self.train_dataset = FUNSDDataset(
                data_dir=self.config['finetuning']['train_dir'],
                max_seq_length=self.config['model']['max_seq_length'],
                label_map=self.config['finetuning']['label_map']
            )
            logger.info(f"Train dataset size: {len(self.train_dataset)}")
            
            logger.info(f"Initializing eval dataset from {self.config['finetuning']['eval_dir']}")
            self.eval_dataset = FUNSDDataset(
                data_dir=self.config['finetuning']['eval_dir'],
                max_seq_length=self.config['model']['max_seq_length'],
                label_map=self.config['finetuning']['label_map']
            )
            logger.info(f"Eval dataset size: {len(self.eval_dataset)}")
        except Exception as e:
            logger.error(f"Error creating datasets: {str(e)}")
            raise

        if len(self.train_dataset) == 0 or len(self.eval_dataset) == 0:
            logger.error("One or both datasets are empty")
            raise ValueError("One or both datasets are empty")

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config['finetuning']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers']
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config['finetuning']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers']
        )
        
        logger.info(f"Train dataloader size: {len(self.train_dataloader)}")
        logger.info(f"Eval dataloader size: {len(self.eval_dataloader)}")

    def setup_model(self):
        logger.info("Setting up model...")
        # Add num_labels to the model config
        self.config['model']['num_labels'] = len(self.config['finetuning']['label_map'])
        
        # Load the pre-trained LayoutLMv3 model
        pretrained_path = self.config['pretraining']['model_path']
        if os.path.exists(pretrained_path):
            logger.info(f"Loading pre-trained model from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            
            # Remove the 'layoutlmv3.' prefix from the keys if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('layoutlmv3.'):
                    new_state_dict[k[11:]] = v
                else:
                    new_state_dict[k] = v
            
            # Create the model
            self.model = LayoutLMv3ForTokenClassification(self.config['model'])
            
            # Load the modified state dict, ignoring the classifier layer
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and 'classifier' not in k}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            
            logger.info("Pre-trained weights loaded successfully")
            
            # Reinitialize the classifier layer
            self.model.classifier = nn.Linear(self.config['model']['hidden_size'], self.config['model']['num_labels'])
            logger.info(f"Classifier layer reinitialized with {self.config['model']['num_labels']} output classes")
        else:
            logger.warning(f"Pre-trained model not found at {pretrained_path}. Initializing from scratch.")
            self.model = LayoutLMv3ForTokenClassification(self.config['model'])

        self.model.to(self.device)
        logger.info("Model initialized and moved to device")

    def setup_optimization(self):
        logger.info("Setting up optimizer and scheduler...")
        self.optimizer = get_optimizer(self.model, self.config['optimizer'])
        total_steps = len(self.train_dataloader) * self.config['finetuning']['num_epochs']
        self.scheduler = get_scheduler(self.optimizer, self.config['scheduler'], total_steps)

    def train(self):
        logger.info("Starting fine-tuning...")
        for epoch in tqdm(range(self.config['finetuning']['num_epochs']), desc="Epochs"):
            self.train_epoch(epoch)
            self.evaluate(epoch)
            self.save_checkpoint(epoch)
        
        self.save_final_model()
        logger.info("Fine-tuning completed")

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        progress_bar = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch+1}/{self.config['finetuning']['num_epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                bbox = batch['bbox'].to(self.device)
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    labels=labels
                )
                loss = outputs['loss']
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                logger.error(traceback.format_exc())
                continue  # Skip this batch and continue with the next one

        avg_loss = epoch_loss / len(self.train_dataloader)
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

    def evaluate(self, epoch):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        eval_progress_bar = tqdm(self.eval_dataloader, desc=f"Evaluating Epoch {epoch+1}/{self.config['finetuning']['num_epochs']}")
        
        with torch.no_grad():
            for batch in eval_progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                bbox = batch['bbox'].to(self.device)
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values
                )
                
                preds = outputs['logits'].argmax(dim=-1)
                all_preds.extend(preds[attention_mask.bool()].cpu().numpy())
                all_labels.extend(labels[attention_mask.bool()].cpu().numpy())

        # Convert label IDs back to label names
        id2label = {v: k for k, v in self.config['finetuning']['label_map'].items()}
        all_preds_labels = [id2label.get(pred, f"unknown_{pred}") for pred in all_preds]
        all_true_labels = [id2label.get(label, f"unknown_{label}") for label in all_labels]

        # Get unique labels from both predictions and true labels
        unique_labels = sorted(set(all_preds_labels + all_true_labels))

        logger.info(f"Unique predicted labels: {set(all_preds_labels)}")
        logger.info(f"Unique true labels: {set(all_true_labels)}")
        logger.info(f"All unique labels: {unique_labels}")

        # Generate classification report
        report = classification_report(all_true_labels, all_preds_labels, labels=unique_labels)
        logger.info(f"Evaluation results for epoch {epoch+1}:\n{report}")

        # Additional analysis
        label_counts = {label: all_true_labels.count(label) for label in unique_labels}
        logger.info(f"Label counts in true labels: {label_counts}")

        pred_counts = {label: all_preds_labels.count(label) for label in unique_labels}
        logger.info(f"Label counts in predictions: {pred_counts}")

    def save_checkpoint(self, epoch):
        if (epoch + 1) % self.config['finetuning']['save_every'] == 0:
            checkpoint_path = os.path.join(self.config['finetuning']['checkpoint_dir'], f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    def save_final_model(self):
        final_model_path = self.config['finetuning']['model_path']
        torch.save(self.model.state_dict(), final_model_path)
        logger.info(f"Final fine-tuned model saved: {final_model_path}")

def finetune_model(config):
    try:
        finetuner = FineTuner(config)
        finetuner.train()
    except Exception as e:
        logger.error(f"An error occurred during fine-tuning: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        finetune_model(config)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
