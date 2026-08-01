from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import io

from model import DualBranchCNN

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DualBranchCNN().to(device)
model.load_state_dict(torch.load('dual_branch_model.pth', map_location=device))
model.eval()

transform = transforms.ToTensor()

def preprocess(image: Image.Image):
    img = image.convert('RGB').resize((32, 32))
    img_tensor = transform(img)

    img_gray = img_tensor.mean(dim=0).numpy()
    fshift = np.fft.fftshift(np.fft.fft2(img_gray))
    magnitude = np.log(np.abs(fshift) + 1e-8)
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    fft_tensor = torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0)

    return img_tensor.unsqueeze(0).to(device), fft_tensor.unsqueeze(0).to(device)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    img_tensor, fft_tensor = preprocess(image)

    with torch.no_grad():
        output = model(img_tensor, fft_tensor)
        probs = F.softmax(output, dim=1)[0]

    return {
        "fake_probability": float(probs[0]),
        "real_probability": float(probs[1]),
        "prediction": "FAKE" if probs[0] > probs[1] else "REAL"
    }