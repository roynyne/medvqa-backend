import io
import json
import torch
from fastapi import APIRouter, File, UploadFile, Form
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from app.model.med_vqa import MedVQA
from app.llm_guard import is_valid_question  # ✅ updated function import

router = APIRouter()

# === Config ===
MODEL_PATH = "saved_model/medvqa_epoch_10.pth"
LABEL_MAP_PATH = "saved_model/label2ans.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model ===
model = MedVQA()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Load tokenizer & label map ===
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
with open(LABEL_MAP_PATH, "r") as f:
    label2ans = json.load(f)

# === Image preprocessing ===
image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


@router.post("/predict")
async def predict(image: UploadFile = File(...), question: str = Form(...)):
    try:
        # My tested LLM guard - only for testing ===
        guard_result = is_valid_question(question)
        if not guard_result.get("valid"):
            return {
                "answer": None,
                "message": guard_result.get("message", "Invalid question.")
            }

        # 1. Load and preprocess image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("L")
        img_tensor = image_transform(pil_image).unsqueeze(0).to(DEVICE)

        # 2. Tokenize question
        inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        # 3. Run inference
        with torch.no_grad():
            logits = model(img_tensor, input_ids, attention_mask)
            predicted_idx = torch.argmax(logits, dim=1).item()
            predicted_answer = label2ans.get(str(predicted_idx), "Unknown")

        return {
            "answer": predicted_answer,
            "class_index": predicted_idx
        }

    except Exception as e:
        return {"error": str(e)}
