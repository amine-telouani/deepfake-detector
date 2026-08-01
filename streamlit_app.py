import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DualBranchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.rgb_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.rgb_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fft_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.fft_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fft_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 4 * 4 * 2, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, rgb, fft):
        r = self.pool(F.relu(self.rgb_conv1(rgb)))
        r = self.pool(F.relu(self.rgb_conv2(r)))
        r = self.pool(F.relu(self.rgb_conv3(r)))
        r = r.view(r.size(0), -1)
        f = self.pool(F.relu(self.fft_conv1(fft)))
        f = self.pool(F.relu(self.fft_conv2(f)))
        f = self.pool(F.relu(self.fft_conv3(f)))
        f = f.view(f.size(0), -1)
        combined = torch.cat([r, f], dim=1)
        x = self.dropout(F.relu(self.fc1(combined)))
        return self.fc2(x)

@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = DualBranchCNN().to(device)
    model.load_state_dict(torch.load('backend/dual_branch_model.pth', map_location=device))
    model.eval()
    return model

def to_tensor(img: Image.Image):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return torch.tensor(arr, dtype=torch.float32)

model = load_model()

st.title("AI-Generated Image Detector")
st.write("Upload an image to check if it's REAL or AI-GENERATED.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded image", use_container_width=True)

    img_tensor = to_tensor(image.resize((32, 32)))

    img_gray = img_tensor.mean(dim=0).numpy()
    fshift = np.fft.fftshift(np.fft.fft2(img_gray))
    magnitude = np.log(np.abs(fshift) + 1e-8)
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    fft_tensor = torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0), fft_tensor.unsqueeze(0))
        probs = F.softmax(output, dim=1)[0]

    fake_prob, real_prob = float(probs[0]), float(probs[1])
    prediction = "FAKE" if fake_prob > real_prob else "REAL"

    st.subheader(f"Prediction: {prediction}")
    st.write(f"Fake: {fake_prob*100:.1f}% | Real: {real_prob*100:.1f}%")