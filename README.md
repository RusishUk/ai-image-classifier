# ai_project/
# ├── model.py       # Код для обучения и сохранения модели
# ├── api.py         # REST API для взаимодействия с моделью
# ├── app.py         # Streamlit UI для тестирования модели
# ├── requirements.txt  # Список зависимостей
# ├── README.md      # Документация проекта
# ├── data/          # Папка с датасетом
# ├── models/        # Папка для сохранения модели
# ├── utils/         # Вспомогательные функции

# model.py (Обучение модели на PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Простая CNN модель
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 28 * 28, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Обучение модели
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(1):  # 1 эпоха для простоты
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Сохранение модели
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/mnist_cnn.pth")

# api.py (REST API на FastAPI)
from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()
model = SimpleCNN()
model.load_state_dict(torch.load("models/mnist_cnn.pth"))
model.eval()

transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    prediction = torch.argmax(output, dim=1).item()
    return {"prediction": prediction}

# app.py (UI на Streamlit)
import streamlit as st
import requests
from PIL import Image

st.title("AI Classifier UI")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:8000/predict", files=files)
        st.write("Prediction:", response.json()["prediction"])

# README.md (Документация)
# AI Image Classifier

## 📌 Описание
Этот проект использует нейросеть на PyTorch для классификации изображений.

## 🚀 Установка
```sh
pip install -r requirements.txt
```

## 🔧 Запуск
### 1. Запуск API
```sh
uvicorn api:app --reload
```
### 2. Запуск UI
```sh
streamlit run app.py

