# ViT-From-Scratch - Vision Transformer on GTSRB

Brief: A PyTorch implementation of the original Vision Transformer (ViT) trained from scratch on the GTSRB traffic-sign dataset, plus a Gradio app that visualizes internals (patches, attention, PCA embedding, Grad-CAM, head diversity).

## Repository structure
- notebooks/ViT_experiemnt.ipynb - training, augmentation, model definition and training loop.
- src/app.py - Gradio app to inspect ViT internals and visualizations.
- models/ - trained model checkpoints (vit_gtsrb_clahe.pth).
- data/ - GTSRB dataset (downloaded by code).
- README.md - this file.

## Dataset
This project uses the GTSRB dataset. The notebooks and app call torchvision.datasets.GTSRB and will download data into `./data`.

## Training
- Main training experiment: `notebooks/ViT_experiemnt.ipynb`.
Key hyperparameters (from notebook):

| Parameter | Value |
|---|---:|
| IMAGE_SIZE | 32 |
| PATCH_SIZE | 4 |
| HIDDEN_SIZE | 384 |
| TRANSFORMER_BLOCKS | 10 |
| ATTENTION_HEADS | 8 |
| BATCH_SIZE | 512 |
| EPOCHS | 80 |
| LR | 6e-4 |
| WEIGHT_DECAY | 0.05 |
| LABEL_SMOOTHING | 0.1 |
| AUGMENTATION | CLAHE, rotation, random crop |
- Run the notebook to train and save model to `vit_gtsrb_clahe.pth`.

## Model (high level)
- PatchEmbedding: Conv2d to create flattened patch tokens.
- TransformerEncoder: LayerNorm -> MultiHeadAttention -> Residual -> MLP with GELU -> Residual.
- CLS token + position embeddings; classification via an MLP head on CLS embedding.

## Visualization app
Path: `src/app.py` - launches a Gradio interface that provides five views:
1. Input (Robot View) - exploded patch grid.
2. Decision (Attention) - CLS attention heatmap overlay.
3. Features (Embedding) - PCA projection vs. background samples from training set.
4. Explainability (Grad-CAM) - Grad-CAM overlay computed on transformer activations.
5. Diversity (Heads) - per-head attention maps.

Run:
- Place trained checkpoint at `models/vit_gtsrb_clahe.pth` (or update path in app).
- python src/testing/app.py

## Screenshots (placeholders)
![Exploded Patches](notes\one.png)
![Attention Heatmap](notes\two.png)
![PCA Projection](notes\three.png)
![Grad-CAM](notes\four.png)
![Head Diversity](notes\five.png)


## License & Acknowledgements
- Original ViT paper: "An Image is Worth 16x16 Words".