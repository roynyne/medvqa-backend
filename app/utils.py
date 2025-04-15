import json
from torchvision import transforms
from transformers import AutoTokenizer

def get_image_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def load_tokenizer():
    return AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

def load_label_mappings(ans_path='app/weights/ans2label.json', label_path='app/weights/label2ans.json'):
    with open(ans_path, 'r') as f:
        ans2label = json.load(f)
    with open(label_path, 'r') as f:
        label2ans = json.load(f)
    return ans2label, label2ans