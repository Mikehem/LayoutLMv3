import logging
import os
import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.data.dataset import PretrainingDataset
from src.model.layoutlmv3 import LayoutLMv3ForPreTraining
from src.utils.helpers import set_seed, get_optimizer, get_scheduler
import traceback
from src.data.prepare_data import prepare_pretraining_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
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
        os.makedirs(self.config['pretraining']['checkpoint_dir'], exist_ok=True)
        
        # Create directory for the final model
        model_dir = os.path.dirname(self.config['pretraining']['model_path'])
        os.makedirs(model_dir, exist_ok=True)

    def setup_data(self):
        logger.info("Setting up data...")
        prepare_pretraining_data(self.config)
        self.dataset = PretrainingDataset(
            data_dir=self.config['data']['dir'],
            max_seq_length=self.config['model']['max_seq_length']
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config['pretraining']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers']
        )
        logger.info(f"Dataset size: {len(self.dataset)}")

    def setup_model(self):
        logger.info("Setting up model...")
        self.model = LayoutLMv3ForPreTraining(self.config['model'])
        self.model.to(self.device)
        logger.info("Model initialized and moved to device")

    def setup_optimization(self):
        logger.info("Setting up optimizer and scheduler...")
        self.optimizer = get_optimizer(self.model, self.config['optimizer'])
        total_steps = len(self.dataloader) * self.config['pretraining']['num_epochs']
        self.scheduler = get_scheduler(self.optimizer, self.config['scheduler'], total_steps)

    def train(self):
        logger.info("Starting training...")
        for epoch in range(self.config['pretraining']['num_epochs']):
            self.train_epoch(epoch)
            self.save_checkpoint(epoch)
        
        self.save_final_model()
        logger.info("Training completed")

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config['pretraining']['num_epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                bbox = batch['bbox'].to(self.device)
                pixel_values = batch['pixel_values'].to(self.device).float()  # Ensure float type
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                logger.error(traceback.format_exc())
                continue  # Skip this batch and continue with the next one

        avg_loss = epoch_loss / len(self.dataloader)
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

    def save_checkpoint(self, epoch):
        if (epoch + 1) % self.config['pretraining']['save_every'] == 0:
            checkpoint_path = os.path.join(self.config['pretraining']['checkpoint_dir'], f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    def save_final_model(self):
        final_model_path = self.config['pretraining']['model_path']
        torch.save(self.model.state_dict(), final_model_path)
        logger.info(f"Final model saved: {final_model_path}")

def train_model(config):
    try:
        trainer = Trainer(config)
        trainer.train()
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
