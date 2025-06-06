import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
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
def load_uploaded_model(model_architecture_name, uploaded_model_file):
    if uploaded_model_file is None:
        return None

    bytes_data = uploaded_model_file.getvalue()
    buffer = io.BytesIO(bytes_data)

    if model_architecture_name == "EfficientNet-B4":
        model = get_efficientnet_b4(num_classes)
    elif model_architecture_name == "ResNet-101":
        model = get_resnet101(num_classes)
    else:
        st.error("Invalid model architecture selected for loading.")
        return None

    try:
        model.load_state_dict(torch.load(buffer, map_location=device))
        model.eval()  
        return model.to(device)
    except Exception as e:
        st.error(
            f"Error loading model from uploaded file. Please ensure it's a valid {model_architecture_name} state_dict. Error: {e}"
        )
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

st.title("Image Classification with Custom PyTorch Models")
st.write("Upload your trained model (`.pth` file) and image(s) for classification.")

model_architecture_choice = st.sidebar.selectbox(
    "1. Choose Your Trained Model Architecture",
    ("EfficientNet-B4", "ResNet-101")
)

uploaded_model_file = st.sidebar.file_uploader(
    f"2. Upload {model_architecture_choice} Model (.pth)",
    type=["pth"]
)

class_names_str = st.sidebar.text_input(
    "3. Enter Class Names (comma-separated)",
    "Nasta Box LIGHT 宅配ボック, Nasta_Box_POST_門柱ユニットタイプ, Nasta_Post_KS-GP10AN_KS-GP10ANKT, Nasta_Post_門柱ユニット KS-GP21A"
)
class_names = [name.strip() for name in class_names_str.split(",")]

model = None
if uploaded_model_file:
    model = load_uploaded_model(model_architecture_choice, uploaded_model_file)
    if model:
        st.sidebar.success(f"{model_architecture_choice} model loaded successfully!")
    else:
        st.sidebar.error("Failed to load model. Please check the file and architecture.")
else:
    st.sidebar.info("Please upload your model file (.pth) to begin.")

if model:
    st.subheader("Upload Image(s) For Classification:")
    uploaded_image_files = st.file_uploader(
        "Drag and drop images here, or click to select",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_image_files:
        st.subheader("Prediction Results:")
        for uploaded_file in uploaded_image_files:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(
                    image, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width =True
                )

                predicted_class, confidence = predict_image(model, image, class_names)

                st.write(f"**Prediction:** {predicted_class}")
                st.write(f"**Confidence:** {confidence:.2f}")
                st.markdown("---")  
            except Exception as e:
                st.error(f"Error processing image {uploaded_file.name}: {e}")
        else:
            st.info("Please upload an image to get a prediction.")
    else:
        st.warning(
            "Please upload a model file and configure class names to start classifying images."
        )