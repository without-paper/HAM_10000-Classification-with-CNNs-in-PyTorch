import os
import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
from models.ResNet_model101 import ResNet101

# 定义模型类别
class_names = {
    'akiec': 'ACTINIC KERATOSIS',
    'bcc': 'BASAL CELL CARCINOMA',
    'bkl': 'BENIGN KERATOSIS-LIKE LESIONS',
    'df': 'DERMATOFIBROMA',
    'mel': 'MELANOMA',
    'nv': 'MELANOCYTIC NEVI',
    'vasc': 'VASCULAR LESIONS'
}
labels = list(class_names.values())

# 图像预处理流程
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7633, 0.5458, 0.5704], std=[0.09, 0.1188, 0.1334])
])

# 加载模型
device = torch.device("cpu")
model = ResNet101(dropout_prob=0.5)
model.load_state_dict(torch.load("pth/resnet101_model.pth", map_location=device))
model.to(device)
model.eval()

# 推理函数
def classify_skin_image(image: Image.Image):
    image = image.convert("RGB")
    tensor = data_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()
        confidence = torch.nn.functional.softmax(output, dim=1)[0][pred].item()
    return {labels[pred]: float(confidence)}

# 构建 Gradio 界面
demo = gr.Interface(
    fn=classify_skin_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Skin Cancer Classifier (ResNet101)",
    description="Upload a skin lesion image and the model will classify it into one of seven categories. The results are for reference only. This model can only classify an image into one of seven skin disease categories, and its accuracy may be affected by various factors.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
