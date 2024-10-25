import torch
import torch.nn as nn
from .layoutlmv3 import LayoutLMv3

class LayoutLMv3ForTokenClassification(nn.Module):
    def __init__(self, config):
        super(LayoutLMv3ForTokenClassification, self).__init__()
        self.num_labels = config['num_labels']
        self.layoutlmv3 = LayoutLMv3(config)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.classifier = nn.Linear(config['hidden_size'], self.num_labels)

    def forward(self, input_ids, attention_mask, bbox, pixel_values, labels=None):
        outputs = self.layoutlmv3(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {'loss': loss, 'logits': logits}

    def freeze_base_model(self):
        """Freeze the parameters of the base LayoutLMv3 model"""
        for param in self.layoutlmv3.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self):
        """Unfreeze the parameters of the base LayoutLMv3 model"""
        for param in self.layoutlmv3.parameters():
            param.requires_grad = True
