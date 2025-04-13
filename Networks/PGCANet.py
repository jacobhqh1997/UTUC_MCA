import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home/hqh/Dataset_2T/UTUC_NCT/')
from Networks.blocks import AttentionNetGated, CombinedPreGatedMultiheadAttention
from Networks.fusion import BilinearFusion, ConcatFusion, GatedConcatFusion
from typing import List

class PreGatedContextualAttentionGateTransformer(nn.Module):
    def __init__(self, mic_sizes: List[int], model_size: str = 'medium', n_classes: int = 4, dropout: float = 0.25, fusion: str = 'concat', device: str = 'cpu'):
        super(PreGatedContextualAttentionGateTransformer, self).__init__()
        self.n_classes = n_classes
        self.model_sizes = [128, 128]
        if model_size == 'small':
            self.model_sizes = [128, 128]
        elif model_size == 'medium':
            self.model_sizes = [256, 256]   
        elif model_size == 'big':
            self.model_sizes = [512, 512]

        # H
        fc = nn.Sequential(
            nn.Linear(1024, self.model_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.H = fc

        # G
        mic_encoders = []
        for mic_size in mic_sizes:
            fc = nn.Sequential(
                nn.Sequential(
                    nn.Linear(mic_size, self.model_sizes[0]),
                    nn.ELU(),
                    nn.AlphaDropout(p=dropout, inplace=False)),
                nn.Sequential(
                    nn.Linear(self.model_sizes[0], self.model_sizes[1]),
                    nn.ELU(),
                    nn.AlphaDropout(p=dropout, inplace=False))
            )
            mic_encoders.append(fc)
        self.G = nn.ModuleList(mic_encoders)

        self.co_attention_M = CombinedPreGatedMultiheadAttention(
            dim1=self.model_sizes[1], 
            dim2=self.model_sizes[1], 
            dk=self.model_sizes[1], 
            output_dim=self.model_sizes[1],  # 与TransformerEncoder的d_model一致
            num_heads=4
        )

        # Path Transformer (T_H)
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_sizes[1], nhead=8, dim_feedforward=512, dropout=dropout,
                                                        activation='relu')
        self.path_transformer_M = nn.TransformerEncoder(path_encoder_layer, num_layers=2)


        # WSI Global Attention Pooling (rho_H_M)
        self.path_attention_head_M = AttentionNetGated(n_classes=1, input_dim=self.model_sizes[1], hidden_dim=self.model_sizes[1])
        self.path_rho_M = nn.Sequential(*[nn.Linear(self.model_sizes[1], self.model_sizes[1]), nn.ReLU(), nn.Dropout(dropout)])

        # macro Transformer (T_G)
        macro_encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_sizes[1], nhead=8, dim_feedforward=512, dropout=dropout,
                                                        activation='relu')
        self.macro_transformer = nn.TransformerEncoder(macro_encoder_layer, num_layers=2)

        # macro Global Attention Pooling (rho_G)
        self.macro_attention_head = AttentionNetGated(n_classes=1, input_dim=self.model_sizes[1], hidden_dim=self.model_sizes[1])
        self.macro_rho = nn.Sequential(*[nn.Linear(self.model_sizes[1], self.model_sizes[1]), nn.ReLU(), nn.Dropout(dropout)])
        
        # Fusion Layer
        self.fusion = fusion
        if self.fusion == 'concat':
            self.fusion_layer = ConcatFusion(dims=[self.model_sizes[1], self.model_sizes[1]],
                                             hidden_size=self.model_sizes[1], output_size=self.model_sizes[1]).to(device=device)
         
        elif self.fusion == 'bilinear':
            self.fusion_layer = BilinearFusion(dim1=self.model_sizes[1], dim2=self.model_sizes[1], output_size=self.model_sizes[1])
        elif self.fusion == 'gated_concat':
            self.fusion_layer = GatedConcatFusion(dims=[self.model_sizes[1], self.model_sizes[1]],
                                                  hidden_size=self.model_sizes[1], output_size=self.model_sizes[1]).to(device=device)
        else:
            raise RuntimeError(f'Fusion mechanism {self.fusion} not implemented')

        # Classifier
        self.classifier = nn.Linear(self.model_sizes[1], n_classes)

    def forward(self, wsi, macros):
        # WSI Fully connected layer
        # H_bag: (Mxd_k)
        
        H_bag = self.H(wsi).squeeze(0)
    
        M_bag = self.G[0](macros.type(torch.float32)).squeeze(0)


        H_coattn_M, M_coattn = self.co_attention_M(H_bag, M_bag)
  
        path_trans_M = self.path_transformer_M(H_coattn_M)
        macro_trans = self.macro_transformer(M_bag)

        # Global Attention Pooling
        A_path_M, h_path_M = self.path_attention_head_M(path_trans_M.squeeze(1))
        A_path_M = torch.transpose(A_path_M, 1, 0)
        h_path_M = torch.mm(F.softmax(A_path_M, dim=1), h_path_M)
        h_path_M = self.path_rho_M(h_path_M).squeeze()

        A_macro, h_macro = self.macro_attention_head(macro_trans.squeeze(1))
        A_macro = torch.transpose(A_macro, 1, 0)
        h_macro = torch.mm(F.softmax(A_macro, dim=1), h_macro)
        h_macro = self.macro_rho(h_macro).squeeze()

        h = self.fusion_layer(h_path_M, h_macro)
        # print(h.shape)
        # logits: classifier output
        # size   --> (1, 4)
        # domain --> R
        logits = self.classifier(h).unsqueeze(0)
        # hazards: probability of patient death in interval j
        # size   --> (1, 4)
        # domain --> [0, 1]
        hazards = torch.sigmoid(logits)
        # survs: probability of patient survival after time t
        # size   --> (1, 4)
        # domain --> [0, 1]
        survs = torch.cumprod(1 - hazards, dim=1)
        # Y: predicted probability distribution
        # size   --> (1, 4)
        # domain --> [0, 1] (probability distribution)
        Y = F.softmax(logits, dim=1)

        attention_scores = {'M_coattn': M_coattn,  'macro':A_macro, 'A_path_M':A_path_M}  
                         


        return hazards, survs, Y, attention_scores

    def get_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


