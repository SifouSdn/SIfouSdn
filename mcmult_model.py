"""
MCMulT: Multi-scale Cooperative Multimodal Transformer
Implementation for CMU-MOSI sentiment analysis

Architecture:
- Multi-scale temporal convolution for capturing different time scales
- Cross-modal attention between audio, visual, and text modalities
- Cooperative fusion mechanism
- Regression head for sentiment prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiScaleTemporalConv(nn.Module):
    """Multi-scale temporal convolution to capture different temporal patterns"""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        # Ensure output channels are evenly divisible
        channels_per_conv = out_channels // len(kernel_sizes)
        self.total_out_channels = channels_per_conv * len(kernel_sizes)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, channels_per_conv, 
                     kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm1d(self.total_out_channels)
        
        # Project to desired output channels if needed
        self.proj = nn.Linear(self.total_out_channels, out_channels) if self.total_out_channels != out_channels else None
        
    def forward(self, x):
        # x: (batch, seq_len, channels)
        x = x.transpose(1, 2)  # (batch, channels, seq_len)
        conv_outputs = [conv(x) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)
        x = self.bn(x)
        x = F.relu(x)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)
        if self.proj is not None:
            x = self.proj(x)
        return x


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism"""
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, query, key, value, attn_mask=None):
        # Self-attention
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attn_mask)
        x = self.norm1(query + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


class CooperativeFusion(nn.Module):
    """Cooperative fusion mechanism for multimodal features"""
    def __init__(self, d_model, num_modalities=3, dropout=0.1):
        super().__init__()
        self.num_modalities = num_modalities
        self.attention_weights = nn.Parameter(torch.ones(num_modalities))
        self.gate = nn.Sequential(
            nn.Linear(d_model * num_modalities, d_model),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
    def forward(self, modality_features):
        # modality_features: list of (batch, seq_len, d_model)
        batch_size = modality_features[0].size(0)
        
        # Normalize attention weights
        weights = F.softmax(self.attention_weights, dim=0)
        
        # Weighted sum of modality features
        weighted_sum = sum(w * feat for w, feat in zip(weights, modality_features))
        
        # Concatenate all modalities
        concat_features = torch.cat(modality_features, dim=-1)
        
        # Gated fusion
        gate_values = self.gate(concat_features)
        fused = weighted_sum * gate_values
        
        return fused


class MCMulT(nn.Module):
    """
    Multi-scale Cooperative Multimodal Transformer
    
    Args:
        audio_dim: Input dimension for audio features
        visual_dim: Input dimension for visual features
        text_dim: Input dimension for text features
        d_model: Hidden dimension for transformer
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
    """
    def __init__(self, audio_dim=74, visual_dim=35, text_dim=300, 
                 d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projections
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.visual_proj = nn.Linear(visual_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        
        # Multi-scale temporal convolutions
        self.audio_conv = MultiScaleTemporalConv(d_model, d_model)
        self.visual_conv = MultiScaleTemporalConv(d_model, d_model)
        self.text_conv = MultiScaleTemporalConv(d_model, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Cross-modal attention layers
        self.cross_modal_layers = nn.ModuleList([
            nn.ModuleDict({
                'audio_to_text': CrossModalAttention(d_model, nhead, dropout),
                'audio_to_visual': CrossModalAttention(d_model, nhead, dropout),
                'visual_to_text': CrossModalAttention(d_model, nhead, dropout),
                'visual_to_audio': CrossModalAttention(d_model, nhead, dropout),
                'text_to_audio': CrossModalAttention(d_model, nhead, dropout),
                'text_to_visual': CrossModalAttention(d_model, nhead, dropout),
            })
            for _ in range(num_layers)
        ])
        
        # Cooperative fusion
        self.fusion = CooperativeFusion(d_model, num_modalities=3, dropout=dropout)
        
        # Output layers
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, audio, visual, text, audio_mask=None, visual_mask=None, text_mask=None):
        """
        Args:
            audio: (batch, seq_len_a, audio_dim)
            visual: (batch, seq_len_v, visual_dim)
            text: (batch, seq_len_t, text_dim)
            audio_mask: (batch, seq_len_a) optional
            visual_mask: (batch, seq_len_v) optional
            text_mask: (batch, seq_len_t) optional
        
        Returns:
            predictions: (batch, 1) sentiment scores
        """
        # Project to common dimension
        audio_feat = self.audio_proj(audio)
        visual_feat = self.visual_proj(visual)
        text_feat = self.text_proj(text)
        
        # Apply multi-scale temporal convolutions
        audio_feat = self.audio_conv(audio_feat)
        visual_feat = self.visual_conv(visual_feat)
        text_feat = self.text_conv(text_feat)
        
        # Add positional encoding
        audio_feat = self.pos_encoder(audio_feat)
        visual_feat = self.pos_encoder(visual_feat)
        text_feat = self.pos_encoder(text_feat)
        
        # Cross-modal attention layers
        for layer in self.cross_modal_layers:
            # Each modality attends to others
            audio_to_text = layer['audio_to_text'](audio_feat, text_feat, text_feat, text_mask)
            audio_to_visual = layer['audio_to_visual'](audio_feat, visual_feat, visual_feat, visual_mask)
            
            visual_to_text = layer['visual_to_text'](visual_feat, text_feat, text_feat, text_mask)
            visual_to_audio = layer['visual_to_audio'](visual_feat, audio_feat, audio_feat, audio_mask)
            
            text_to_audio = layer['text_to_audio'](text_feat, audio_feat, audio_feat, audio_mask)
            text_to_visual = layer['text_to_visual'](text_feat, visual_feat, visual_feat, visual_mask)
            
            # Update features with cross-modal information
            audio_feat = (audio_feat + audio_to_text + audio_to_visual) / 3
            visual_feat = (visual_feat + visual_to_text + visual_to_audio) / 3
            text_feat = (text_feat + text_to_audio + text_to_visual) / 3
        
        # Cooperative fusion
        fused_feat = self.fusion([audio_feat, visual_feat, text_feat])
        
        # Global pooling
        fused_feat = fused_feat.transpose(1, 2)  # (batch, d_model, seq_len)
        pooled = self.pooling(fused_feat).squeeze(-1)  # (batch, d_model)
        
        # Classification
        output = self.classifier(pooled)
        
        return output


def create_model(config):
    """Factory function to create MCMulT model from config"""
    model = MCMulT(
        audio_dim=config.get('audio_dim', 74),
        visual_dim=config.get('visual_dim', 35),
        text_dim=config.get('text_dim', 300),
        d_model=config.get('d_model', 128),
        nhead=config.get('nhead', 8),
        num_layers=config.get('num_layers', 4),
        dropout=config.get('dropout', 0.1)
    )
    return model


if __name__ == '__main__':
    # Test model
    batch_size = 4
    seq_len_a, seq_len_v, seq_len_t = 50, 50, 50
    
    audio = torch.randn(batch_size, seq_len_a, 74)
    visual = torch.randn(batch_size, seq_len_v, 35)
    text = torch.randn(batch_size, seq_len_t, 300)
    
    model = MCMulT()
    output = model(audio, visual, text)
    
    print(f"Input shapes: audio={audio.shape}, visual={visual.shape}, text={text.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
