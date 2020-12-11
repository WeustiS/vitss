import torch
from vit_pytorch import ViT

v = ViT(
    image_size = 320,
    patch_size = 16,
    num_classes = 250,
    dim = 1024,
    depth = 2,
    heads = 6,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 320, 320)
mask = torch.ones(1, 20, 20).bool() # optional mask, designating which patch to attend to

preds = v(img, mask = mask) # (1, 1000)
print(preds)