# обучаемые головы поверх замороженных UNI-эмбеддингов

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    # базовая линия без пространственного контекста
    def __init__(self, in_dim, hidden=256, n_classes=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x, edge_index=None):   # edge_index не используется, для единого интерфейса
        return self.net(x)


class GraphSAGEHead(nn.Module):
    def __init__(self, in_dim, hidden=256, n_classes=3, num_layers=2, dropout=0.3):
        super().__init__()
        # torch_geometric импортируем здесь, а не наверху: с head: mlp он не нужен,
        # а тянет за собой пол-гигабайта зависимостей
        from torch_geometric.nn import SAGEConv
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(num_layers)])
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x, edge_index):
        x = self.proj(self.norm(x))
        for conv in self.convs:
            x = F.gelu(conv(x, edge_index)) + x   
            x = F.dropout(x, self.dropout, training=self.training)
        return self.head(x)

    def freeze_first_layer(self):
        for p in self.proj.parameters():
            p.requires_grad = False
        for p in self.convs[0].parameters():
            p.requires_grad = False


def build_head(cfg_train, in_dim, n_classes=3):
    h = cfg_train["head"].lower()
    if h == "mlp":
        return MLPHead(in_dim, cfg_train["hidden_dim"], n_classes, cfg_train["dropout"])
    if h == "graphsage":
        return GraphSAGEHead(in_dim, cfg_train["hidden_dim"], n_classes,
                             cfg_train["num_layers"], cfg_train["dropout"])
    raise ValueError(f"неизвестная голова: {h}")