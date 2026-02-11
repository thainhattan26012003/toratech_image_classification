"""
Backend API for Image Classification - port 1234
Upload PyTorch model (.pth) and classify images.
"""
import io
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms

num_classes = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# In-memory model state (single model at a time)
current_model = None
current_class_names = []
current_architecture = None


def get_efficientnet_b4(n_classes):
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    return model


def get_resnet101(n_classes):
    model = models.resnet101(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def load_model_from_bytes(bytes_data: bytes, architecture: str, class_names: list):
    buffer = io.BytesIO(bytes_data)
    if architecture == "EfficientNet-B4":
        model = get_efficientnet_b4(num_classes)
    elif architecture == "ResNet-101":
        model = get_resnet101(num_classes)
    else:
        raise ValueError(f"Invalid architecture: {architecture}")
    model.load_state_dict(torch.load(buffer, map_location=device))
    model.eval()
    return model.to(device)


def predict_image(model, image: Image.Image, class_names: list):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_class_idx = torch.max(probabilities, 0)
    predicted_class_name = class_names[predicted_class_idx.item()]
    confidence_score = confidence.item()
    return predicted_class_name, confidence_score


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # cleanup if needed
    global current_model
    current_model = None


app = FastAPI(title="Image Classification API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
@app.head("/")
def root():
    """Root endpoint for load balancers / health probes that hit /."""
    return {"service": "image-classification-api", "status": "ok"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/model")
async def upload_model(
    architecture: str = Form(...),
    class_names: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload and load a trained model (.pth)."""
    global current_model, current_class_names, current_architecture
    if not file.filename or not file.filename.lower().endswith(".pth"):
        raise HTTPException(status_code=400, detail="File must be a .pth file")
    if architecture not in ("EfficientNet-B4", "ResNet-101"):
        raise HTTPException(status_code=400, detail="Invalid architecture")
    names = [n.strip() for n in class_names.split(",") if n.strip()]
    if len(names) != num_classes:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {num_classes} class names (comma-separated)",
        )
    try:
        data = await file.read()
        current_model = load_model_from_bytes(data, architecture, names)
        current_class_names = names
        current_architecture = architecture
        return {
            "message": f"{architecture} model loaded successfully",
            "class_names": current_class_names,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {str(e)}")


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Classify one image. Returns predicted class and confidence."""
    global current_model, current_class_names
    if current_model is None:
        raise HTTPException(status_code=400, detail="No model loaded. Upload a model first.")
    allowed = ("image/jpeg", "image/jpg", "image/png")
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="File must be JPEG or PNG")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        predicted_class, confidence = predict_image(current_model, image, current_class_names)
        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/api/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Classify multiple images."""
    global current_model, current_class_names
    if current_model is None:
        raise HTTPException(status_code=400, detail="No model loaded. Upload a model first.")
    results = []
    for f in files:
        if not f.filename:
            continue
        if f.content_type not in ("image/jpeg", "image/jpg", "image/png"):
            results.append({"filename": f.filename, "error": "Invalid image type"})
            continue
        try:
            contents = await f.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            predicted_class, confidence = predict_image(current_model, image, current_class_names)
            results.append({
                "filename": f.filename,
                "prediction": predicted_class,
                "confidence": round(confidence, 4),
            })
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1234)
