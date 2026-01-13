import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import GTSRB
import numpy as np
import cv2
import gradio as gr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import os

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
MEAN = (0.3337, 0.3064, 0.3171)
STD = (0.2672, 0.2564, 0.2629)
IMAGE_SIZE = 32
PATCH_SIZE = 4
HIDDEN_SIZE = 384
TRANSFORMER_BLOCKS = 10
ATTENTION_HEADS = 8
MLP_DIMENSION = 4 * HIDDEN_SIZE
NUM_CLASSES = 43
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mapping of GTSRB IDs to Class Names
CLASSES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)', 
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons', 
    11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield', 
    14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited', 
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left', 
    20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road', 
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work', 
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing', 
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing', 
    32: 'End of all speed and passing limits', 33: 'Turn right ahead', 
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right', 
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left', 
    40: 'Roundabout mandatory', 41: 'End of no passing', 
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# ==========================================
# 2. MODEL DEFINITION
# ==========================================

class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, HIDDEN_SIZE, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(HIDDEN_SIZE)
        self.layer_norm2 = nn.LayerNorm(HIDDEN_SIZE)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=HIDDEN_SIZE, 
            num_heads=ATTENTION_HEADS, 
            batch_first=True,
            dropout=dropout
        )
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, MLP_DIMENSION),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(MLP_DIMENSION, HIDDEN_SIZE),
            nn.Dropout(dropout)
        )
        self.attn_weights = None 

    def forward(self, x):
        x_norm = self.layer_norm1(x)
        attention_output, attn_weights = self.multihead_attention(x_norm, x_norm, x_norm, need_weights=True, average_attn_weights=False)
        self.attn_weights = attn_weights 
        x = x + attention_output
        x_norm = self.layer_norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output
        return x

class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.fc(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding()
        self.cls_token = nn.Parameter(torch.randn(1, 1, HIDDEN_SIZE))
        self.position_embeddings = nn.Parameter(torch.randn(1, NUM_PATCHES + 1, HIDDEN_SIZE))
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.Sequential(*[TransformerEncoder(dropout) for _ in range(TRANSFORMER_BLOCKS)])
        self.mlp_head = MLPHead()

    def forward(self, x):
        x = self.patch_embedding(x)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        embedding = x[:, 0, :]
        logits = self.mlp_head(embedding)
        return logits, embedding

# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================

def apply_clahe(img):
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final_img

def load_model():
    model = VisionTransformer()
    try:
        state_dict = torch.load(r'C:\Users\Lenovo\Documents\projects\personal\ViT-From-Scratch\models\vit_gtsrb_clahe.pth', map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("✅ Model loaded successfully.")
    except FileNotFoundError:
        print("⚠️ WARNING: .pth file not found. Visualizations will be random.")
    
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image):
    """Prepares image for the model (32x32)"""
    if image is None: return None
    img_resized = cv2.resize(np.array(image), (32, 32))
    img_clahe = apply_clahe(img_resized)
    img_tensor = transforms.ToTensor()(img_clahe)
    img_tensor = transforms.Normalize(MEAN, STD)(img_tensor)
    return img_tensor.unsqueeze(0).to(DEVICE) 

def overlay_heatmap(heatmap, original_image):
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(original_image), 0.6, heatmap, 0.4, 0)
    return overlay

# ==========================================
# 4. LATENT SPACE GENERATION (REAL DATA)
# ==========================================

# Global storage for PCA and background points
pca_model = None
bg_embeddings_2d = None
bg_labels = None

def generate_latent_space(model):
    """
    Fetches GTSRB data, extracts embeddings for ~20 images per class,
    fits a PCA model, and stores the 2D background map.
    """
    global pca_model, bg_embeddings_2d, bg_labels
    
    print("⏳ Generating Latent Space Map from Training Data (this may take a minute)...")
    
    # Same transform as validation
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Lambda(lambda x: apply_clahe(x)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    # Load Dataset (Download if needed)
    try:
        train_dataset = GTSRB(root='./data', split='train', download=True, transform=transform)
    except Exception as e:
        print(f"❌ Error loading GTSRB: {e}")
        return

    # Select ~20 images per class
    indices = []
    class_counts = {i: 0 for i in range(NUM_CLASSES)}
    MAX_PER_CLASS = 20
    
    # We scan the dataset until we fill our quota
    # (Doing a simple scan is faster than finding all indices first)
    for idx, (_, label) in enumerate(train_dataset):
        if class_counts[label] < MAX_PER_CLASS:
            indices.append(idx)
            class_counts[label] += 1
        
        # Stop if we have enough for all classes (approx check)
        if len(indices) >= NUM_CLASSES * MAX_PER_CLASS:
            break
            
    subset = Subset(train_dataset, indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False)
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            _, embeddings = model(imgs) # (B, 384)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(lbls.numpy())
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    bg_labels = np.array(all_labels)
    
    # Fit PCA
    print(f"🔹 Fitting PCA on {len(all_embeddings)} samples...")
    pca_model = PCA(n_components=2)
    bg_embeddings_2d = pca_model.fit_transform(all_embeddings)
    
    print("✅ Latent Space Generation Complete.")

# ==========================================
# 5. VISUALIZATION LOGIC
# ==========================================

def vis_input_patches(image):
    """Tab 1: Exploded Patch View"""
    img_arr = np.array(image)
    small_img = cv2.resize(img_arr, (32, 32))
    
    H, W, C = small_img.shape
    patch_size = 4
    grid_size = 8
    gap = 2
    canvas_size = grid_size * (patch_size + gap)
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
    
    for i in range(grid_size):
        for j in range(grid_size):
            y = i * patch_size
            x = j * patch_size
            patch = small_img[y:y+patch_size, x:x+patch_size]
            cy = i * (patch_size + gap)
            cx = j * (patch_size + gap)
            canvas[cy:cy+patch_size, cx:cx+patch_size] = patch
            
    canvas = cv2.resize(canvas, (400, 400), interpolation=cv2.INTER_NEAREST)
    return canvas

def vis_attention_map(image_tensor, original_image):
    """Tab 2: Attention Rollout"""
    with torch.no_grad():
        logits, _ = model(image_tensor)
    
    attn = model.transformer_blocks[-1].attn_weights 
    attn_avg = attn.mean(dim=1).squeeze(0)
    cls_attn = attn_avg[0, 1:]
    
    grid_size = int(np.sqrt(NUM_PATCHES))
    attn_map = cls_attn.reshape(grid_size, grid_size).cpu().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    
    return overlay_heatmap(attn_map, original_image)

def vis_embedding_projection(image_tensor):
    """Tab 3: PCA Projection with Real Background"""
    with torch.no_grad():
        _, embedding = model(image_tensor) # (1, 384)
    
    emb_np = embedding.cpu().numpy()
    
    # Project new image using the fitted PCA
    if pca_model is not None:
        projected_point = pca_model.transform(emb_np) # (1, 2)
    else:
        # Fallback if dataset download failed
        projected_point = np.zeros((1, 2))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot Background (Real GTSRB Data)
    if bg_embeddings_2d is not None:
        scatter = ax.scatter(bg_embeddings_2d[:, 0], bg_embeddings_2d[:, 1], 
                             c=bg_labels, cmap='tab20', alpha=0.4, s=15, label='GTSRB Training Samples')
    
    # Plot Input Image
    ax.scatter(projected_point[:, 0], projected_point[:, 1], 
               c='red', s=250, marker='*', edgecolors='black', linewidths=1.5, zorder=10, label='Your Upload')
    
    ax.set_title("Visual Semantic Space (PCA)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    
    # === FIX START ===
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img_plot = data.reshape((h, w, 4))
    img_plot = img_plot[:, :, :3] # Drop Alpha channel to get RGB
    # === FIX END ===
    
    plt.close()
    return img_plot

def vis_gradcam(image_tensor, original_image):
    """Tab 4: Grad-CAM"""
    target_layer = model.transformer_blocks[-1].layer_norm2
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
        
    def forward_hook(module, input, output):
        activations.append(output)
        
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    handle_f = target_layer.register_forward_hook(forward_hook)
    
    model.zero_grad()
    logits, _ = model(image_tensor)
    pred_idx = logits.argmax(dim=1).item()
    logits[0, pred_idx].backward()
    
    handle_b.remove()
    handle_f.remove()
    
    grads = gradients[0][0] 
    acts = activations[0][0]
    weights = torch.mean(grads, dim=0)
    acts_patches = acts[1:, :]
    cam = torch.matmul(acts_patches, weights)
    cam = F.relu(cam)
    
    grid_size = int(np.sqrt(NUM_PATCHES))
    cam = cam.reshape(grid_size, grid_size).detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    return overlay_heatmap(cam, original_image)

def vis_head_diversity(image_tensor, original_image):
    """Tab 5: Head Diversity"""
    with torch.no_grad():
        _ = model(image_tensor)
    
    attn = model.transformer_blocks[-1].attn_weights
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    grid_size = int(np.sqrt(NUM_PATCHES))
    
    for i in range(ATTENTION_HEADS):
        head_attn = attn[0, i, 0, 1:]
        head_map = head_attn.reshape(grid_size, grid_size).cpu().numpy()
        head_map = (head_map - head_map.min()) / (head_map.max() - head_map.min() + 1e-8)
        head_map_resized = cv2.resize(head_map, (original_image.shape[1], original_image.shape[0]))
        
        axes[i].imshow(original_image)
        axes[i].imshow(head_map_resized, cmap='jet', alpha=0.5)
        axes[i].set_title(f"Head {i+1}")
        axes[i].axis('off')
        
    plt.tight_layout()
    
    # === FIX START ===
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img_plot = data.reshape((h, w, 4))
    img_plot = img_plot[:, :, :3] # Drop Alpha channel to get RGB
    # === FIX END ===

    plt.close()
    return img_plot

# ==========================================
# 6. APP SETUP
# ==========================================

# Initialize Model
model = load_model()

# Initialize Latent Space (Runs once at startup)
generate_latent_space(model)

def run_analysis(image):
    if image is None: return None, None, None, None, None, "Please upload an image."
    
    orig_img = np.array(image)
    img_tensor = preprocess_image(image)
    
    with torch.no_grad():
        logits, _ = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[0, pred_idx].item()
        
    pred_text = f"Prediction: {CLASSES[pred_idx]} ({confidence*100:.1f}%)"
    
    vis1 = vis_input_patches(image)
    vis2 = vis_attention_map(img_tensor, orig_img)
    vis3 = vis_embedding_projection(img_tensor)
    vis4 = vis_gradcam(img_tensor, orig_img)
    vis5 = vis_head_diversity(img_tensor, orig_img)
    
    return vis1, vis2, vis3, vis4, vis5, pred_text

# Define Interface
with gr.Blocks(title="ViT Internals Inspector") as demo:
    gr.Markdown("# 👁️ Vision Transformer (ViT) Inspector for GTSRB")
    gr.Markdown("Upload a traffic sign image to see how the Transformer processes it layer by layer.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Upload Image")
            run_btn = gr.Button("Analyze Image", variant="primary")
            prediction_label = gr.Label(label="Model Prediction")
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("1. Input (Robot View)"):
                    gr.Markdown("**Patch Grid:** How the model chops the image into sequences.")
                    out_vis1 = gr.Image(label="Exploded Patches")
                
                with gr.TabItem("2. Decision (Attention)"):
                    gr.Markdown("**Attention Map:** Where the CLS token looked to make the decision.")
                    out_vis2 = gr.Image(label="Attention Heatmap")
                    
                with gr.TabItem("3. Features (Embedding)"):
                    gr.Markdown("**Latent Space (PCA):** Your image (Star) vs 800+ real training images (Dots).")
                    out_vis3 = gr.Image(label="PCA Projection")
                
                with gr.TabItem("4. Explainability (Grad-CAM)"):
                    gr.Markdown("**Grad-CAM:** Gradients showing features that positively contributed to the class.")
                    out_vis4 = gr.Image(label="Grad-CAM Overlay")
                    
                with gr.TabItem("5. Diversity (Heads)"):
                    gr.Markdown("**Head Diversity:** What each specific attention head is focusing on.")
                    out_vis5 = gr.Image(label="Multi-Head Grid")

    run_btn.click(
        run_analysis, 
        inputs=[input_img], 
        outputs=[out_vis1, out_vis2, out_vis3, out_vis4, out_vis5, prediction_label]
    )

if __name__ == "__main__":
    demo.launch()