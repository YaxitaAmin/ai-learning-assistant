# src/core/multimodal_fusion.py
class MultiModalFusion(nn.Module):
    def __init__(self, text_dim=512, vision_dim=512, fusion_dim=768):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + vision_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, text_emb, vision_emb):
        combined = torch.cat([text_emb, vision_emb], dim=1)
        return self.fusion(combined)