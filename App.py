import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

CLASS_NAMES = [
    'battery', 'biological', 'brown-glass', 'cardboard',
    'clothes', 'green-glass', 'metal', 'paper',
    'plastic', 'shoes', 'trash', 'white-glass'
]

THRESHOLD = 0.4

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

@st.cache_resource
def load_model():
    model = CNN(len(CLASS_NAMES))
    model.load_state_dict(
        torch.load(
            "/home/vulethanh/project/Machine Learning/best_model_v3.pth",
            map_location="cpu",
            weights_only=True
        )
    )
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

st.title("♻️ Hệ thống phân loại rác")

uploaded_files = st.file_uploader(
    "Chọn ảnh rác",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption=uploaded_file.name, width=250)

        with col2:
            img = transform(image).unsqueeze(0)

            with torch.no_grad():
                output     = model(img)
                probs      = torch.softmax(output, dim=1)
                confidence, pred = torch.max(probs, dim=1)

            if confidence.item() < THRESHOLD:
                st.warning("⚠️ Không nhận ra đây là rác!")
            else:
                result = CLASS_NAMES[pred.item()]
                st.success(f"**{result.upper()}**")
                st.write(f"Độ tin cậy: **{confidence.item()*100:.2f}%**")

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(CLASS_NAMES, probs.squeeze().numpy(), color="steelblue")
            ax.set_ylabel("Xác suất")
            ax.set_title("Phân phối xác suất")
            plt.xticks(rotation=45, fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)