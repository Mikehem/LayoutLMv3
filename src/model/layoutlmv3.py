import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class VisionEncoder(nn.Module):
    def __init__(
        self,
        img_size=224,        # Size of the input image (224x224 pixels)
        patch_size=16,       # Size of each patch the image is divided into (16x16 pixels)
        in_channels=3,       # Number of color channels in the input image (3 for RGB)
        embed_dim=768,       # Dimension of the embedding space
        depth=12,            # Number of transformer layers
        num_heads=12,        # Number of attention heads in each transformer layer
        mlp_ratio=4.0,       # Ratio for determining the size of the MLP layer
        dropout=0.1,         # Dropout rate for regularization
        attn_dropout=0.1     # Dropout rate specifically for attention layers
    ):
        super(VisionEncoder, self).__init__()
        
        # Ensure the image size is divisible by the patch size
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # Total number of patches
        self.embed_dim = embed_dim

        # Patch Embedding Layer: Converts image patches to embeddings
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Class Token: A learnable vector added to the sequence of embedded patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position Embeddings: Learnable vectors added to patch embeddings to retain positional information
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Dropout layer for regularization
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer Encoder Layers: Process the sequence of patch embeddings
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
            ) for _ in range(depth)
        ])

        # Layer Normalization: Normalizes the output of the transformer layers
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, in_channels, img_size, img_size)
        batch_size = x.size(0)

        # Patch Embedding: Convert image to patch embeddings
        x = self.patch_embed(x)  # Shape: (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)

        # Add class token to the sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: (batch_size, num_patches + 1, embed_dim)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through Transformer Encoder layers
        for layer in self.transformer_encoder:
            x = layer(x)

        # Apply Layer Normalization
        x = self.norm(x)

        # Extract the visual features from the class token
        visual_features = x[:, 0]  # Shape: (batch_size, embed_dim)

        return visual_features

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, attn_dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head Self-Attention layer
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout)
        
        # First layer of the MLP
        self.linear1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Second layer of the MLP
        self.linear2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)

        # Layer Normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout layers for each sub-layer
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation function
        self.activation = nn.GELU()

    def forward(self, src):
        # Self-Attention
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward Network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TextEncoder(nn.Module):
    def __init__(self, config):
        super(TextEncoder, self).__init__()
        # Use the config dictionary directly instead of accessing __dict__
        self.bert = BertModel(BertConfig(**config))

    def forward(self, input_ids, attention_mask):
        textual_features = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return textual_features.last_hidden_state

class SpatialEmbeddings(nn.Module):
    def __init__(self, hidden_size, max_position_embeddings=1000):
        super(SpatialEmbeddings, self).__init__()
        # Embedding layers for each coordinate of the bounding box
        self.x_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.y_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.h_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.w_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

    def forward(self, bbox):
        # Clamp bbox values to be within the embedding range
        bbox = bbox.clamp(min=0, max=999)
        # Convert to long tensor for embedding lookup
        bbox = bbox.long()
        
        # Get embeddings for each coordinate
        x_emb = self.x_embeddings(bbox[:, :, 0])  # x-coordinate
        y_emb = self.y_embeddings(bbox[:, :, 1])  # y-coordinate
        h_emb = self.h_embeddings(bbox[:, :, 2])  # height
        w_emb = self.w_embeddings(bbox[:, :, 3])  # width
        
        # Combine all embeddings
        return x_emb + y_emb + h_emb + w_emb

class LayoutLMv3(nn.Module):
    def __init__(self, config):
        super(LayoutLMv3, self).__init__()
        self.config = config
        self.num_labels = config.get('num_labels', 2)  # Default to 2 if not specified
        self.text_encoder = TextEncoder(config)
        self.vision_encoder = VisionEncoder(
            img_size=224,
            patch_size=16,
            in_channels=3,
            embed_dim=config['hidden_size'],
            depth=config['num_hidden_layers'],
            num_heads=config['num_attention_heads'],
            mlp_ratio=4.0,
            dropout=config['hidden_dropout_prob'],
            attn_dropout=config['attention_probs_dropout_prob']
        )
        self.spatial_embeddings = SpatialEmbeddings(hidden_size=config['hidden_size'])
        self.multimodal_encoder = BertModel(BertConfig(**config))
        self.classifier = nn.Linear(config['hidden_size'], self.num_labels)

    def forward(self, input_ids, attention_mask, bbox, images, labels=None):
        # Handle 3D input tensors
        if input_ids.dim() == 3:
            batch_size, seq_len, num_tokens = input_ids.size()
            input_ids = input_ids.view(batch_size * seq_len, num_tokens)
            attention_mask = attention_mask.view(batch_size * seq_len, num_tokens)
            bbox = bbox.view(batch_size * seq_len, *bbox.size()[2:])
        
        # Ensure images is 4D (batch_size, channels, height, width)
        if images.dim() == 3:
            images = images.unsqueeze(0)

        textual_features = self.text_encoder(input_ids, attention_mask)
        visual_features = self.vision_encoder(images)
        spatial_features = self.spatial_embeddings(bbox)
        
        # Combine features
        combined_features = textual_features + spatial_features + visual_features.unsqueeze(1)
        outputs = self.multimodal_encoder(inputs_embeds=combined_features, attention_mask=attention_mask)
        
        sequence_output = outputs.last_hidden_state
        
        return (sequence_output,)  # Return a tuple with sequence_output

class LayoutLMv3ForPreTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize the main LayoutLMv3 model
        self.layoutlmv3 = LayoutLMv3(config)
        
        # Add a prediction head for masked language modeling
        self.mlm_head = nn.Linear(config['hidden_size'], config['vocab_size'])

    def forward(self, input_ids, attention_mask, bbox, pixel_values, labels=None):
        # Process inputs through the main model
        outputs = self.layoutlmv3(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            images=pixel_values
        )

        sequence_output = outputs[0]  # Get the sequence output
        
        # Generate prediction scores for each token in the vocabulary
        prediction_scores = self.mlm_head(sequence_output)

        # Calculate loss if labels are provided (for training)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(prediction_scores.view(-1, self.layoutlmv3.config['vocab_size']), labels.view(-1))

        return LayoutLMv3PreTrainingOutput(
            loss=loss,
            prediction_scores=prediction_scores,
            hidden_states=None,  # We're not returning hidden states in this implementation
            attentions=None,  # We're not returning attentions in this implementation
        )

class LayoutLMv3PreTrainingOutput:
    def __init__(self, loss, prediction_scores, hidden_states, attentions):
        self.loss = loss
        self.prediction_scores = prediction_scores
        self.hidden_states = hidden_states
        self.attentions = attentions

class LayoutLMv3ForTokenClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config['num_labels']
        self.layoutlmv3 = LayoutLMv3(config)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.classifier = nn.Linear(config['hidden_size'], self.num_labels)

    def forward(self, input_ids, attention_mask, bbox, pixel_values, labels=None):
        outputs = self.layoutlmv3(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            images=pixel_values
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
