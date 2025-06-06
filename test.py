import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import io

num_classes = 4 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_efficientnet_b4(num_classes):
    model = models.efficientnet_b4(weights=None) 
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def get_resnet101(num_classes):
    model = models.resnet101(weights=None) 
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@st.cache_resource 
def load_model(model_name, num_classes, model_path):
    if model_name == "EfficientNet-B4":
        model = get_efficientnet_b4(num_classes)
    elif model_name == "ResNet-101":
        model = get_resnet101(num_classes)
    else:
        st.error("Invalid model name selected.")
        return None

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() 
    except FileNotFoundError:
        st.error(f"Error: Model file not found at '{model_path}'. Please ensure the model is saved.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_image(model, image, class_names):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_class_idx = torch.max(probabilities, 0)

    predicted_class_name = class_names[predicted_class_idx.item()]
    confidence_score = confidence.item()

    return predicted_class_name, confidence_score

st.set_page_config(page_title="Image Classification App", layout="centered")

st.title("Image Classification with PyTorch Models")
st.write("Upload an image (or multiple images) and get predictions using a pre-trained model.")

model_choice = st.sidebar.selectbox(
    "Choose a Model",
    ("EfficientNet-B4", "ResNet-101")
)

class_names = ['Nasta Box LIGHT 宅配ボック', 'Nasta_Box_POST_門柱ユニットタイプ', 'Nasta_Post_KS-GP10AN_KS-GP10ANKT', 'Nasta_Post_門柱ユニット KS-GP21A'] 

st.sidebar.write("---")
st.sidebar.header("Instructions")
st.sidebar.info(
    "1. Choose a model from the dropdown.\n"
    "2. Make sure the corresponding model file (`.pth` or `.pt`) is in the same directory as this script.\n"
    "3. Upload one or more images.\n"
    "4. The app will predict the class and confidence for each image."
)

if model_choice == "EfficientNet-B4":
    model_path = "efficientnet_b4_model.pth" 
elif model_choice == "ResNet-101":
    model_path = "resnet101_model.pth" 

model = load_model(model_choice, num_classes, model_path)

if model:
    st.success(f"{model_choice} loaded successfully!")

    uploaded_files = st.file_uploader(
        "Upload Image(s) for Classification",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.subheader("Prediction Results:")
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)

            predicted_class, confidence = predict_image(model, image, class_names)

            st.write(f"**Prediction:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}")
            st.markdown("---") 
    else:
        st.info("Please upload an image to get a prediction.")
else:
    st.warning("Model could not be loaded. Please check the model path and ensure the file exists.")