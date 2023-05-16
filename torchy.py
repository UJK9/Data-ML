import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json

# Load the pre-trained ResNet model
resnet = models.resnet50()

# Set the model to evaluation mode
resnet.eval()

# Preprocess the input image
image_path = 'Everest.jpg'
image = Image.open(image_path)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0)

# Check if a GPU is available and move the input batch to the GPU if applicable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_batch = input_batch.to(device)

# Make predictions
with torch.no_grad():
    input_batch = input_batch.to(device)
    output = resnet(input_batch)

# Load the class labels
LABELS_FILE= 'classification_classes_ILSVRC2012.txt'
with open(LABELS_FILE, 'r') as f:
    labels= f.read().splitlines()
label_idx= torch.argmax(output)

# Get the predicted label
predicted_label = labels[label_idx.item()]

# Print the predicted label
print("Predicted label:", predicted_label)

