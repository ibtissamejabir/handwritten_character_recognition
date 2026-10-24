import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from streamlit_drawable_canvas import st_canvas

# ------------------------------------------------------------------
# MODEL ARCHITECTURE
# ------------------------------------------------------------------
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin,  cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )

class CharacterCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256 * 3 * 3, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))

# ------------------------------------------------------------------
# LOAD MODEL & CHECKPOINT
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("/Users/Zhuanz/Desktop/handwritten_character_recognition/emnist_cnn_case_merged_best.pth", map_location=device)
    NUM_CLASSES = len(checkpoint["classes"])
    model = CharacterCNN(NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["classes"], device

model, CLASSES, device = load_model()

# ------------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------------
st.title("Handwritten Character Recognizer (90.8%)")
st.markdown(
    "Draw **one** character with a thick stroke below.\n\n"
    "• Labels like **O/o**, **S/s**, **L/l** mean upper and lower case were merged[cite: 3].\n"
    "• When the model is unsure, you will see alternative predictions with similar percentages."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Drawing Canvas")
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=14,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=280,
        height=280,
        drawing_mode="freedraw",
        return_image_data=True,  # <--- Added this parameter
        key="canvas",
    )

def preprocess(gray: np.ndarray) -> np.ndarray | None:
    if gray is None or gray.size == 0:
        return None
    img = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img

if canvas_result.image_data is not None:
    img_data = canvas_result.image_data.astype(np.uint8)
    
    if img_data.ndim == 3:
        if img_data.shape[2] == 4 and img_data[..., 3].min() < 255:
            gray = img_data[..., 3]
        else:
            gray = img_data[..., :3].mean(axis=-1)
    else:
        gray = img_data

    if gray.mean() > 127:
        gray = 255 - gray

    if gray.mean() >= 1:
        x = preprocess(gray)
        if x is not None:
            tensor_x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor_x)
                probs = F.softmax(logits, dim=1).squeeze(0)

            top_probs, top_indices = torch.topk(probs, k=5)

            with col2:
                st.subheader("What the model sees (28×28)")
                preview = (x * 255).astype(np.uint8)
                preview_resized = cv2.resize(preview, (140, 140), interpolation=cv2.INTER_NEAREST)
                st.image(preview_resized, width=140)

                st.subheader("Top Predictions")
                for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                    cls = CLASSES[idx]
                    pct = float(prob) * 100
                    bar = "█" * int(pct / 5)
                    st.text(f"{cls:<10} {pct:5.1f}%  {bar}")