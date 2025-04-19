#!/usr/bin/env python3
"""
Simple bean classification app using the Hugging Face transformers library.
"""

# Import statements
print("Importing libraries...")
import os
import sys
import torch
import gradio as gr
from PIL import Image
import numpy as np

# Print Python and package versions for debugging
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"PIL version: {Image.__version__}")

# Import transformers
print("Importing transformers...")
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# Define a simple function to classify beans
def classify_bean(image):
    print(f"Received image type: {type(image)}")
    
    if image is None:
        print("No image provided")
        return {"angular_leaf_spot": 0.0, "bean_rust": 0.0, "healthy": 0.0}
    
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            print("Converting numpy array to PIL Image")
            image = Image.fromarray(image.astype('uint8')).convert("RGB")
        
        # Ensure image is in RGB mode
        if image.mode != "RGB":
            print(f"Converting image from {image.mode} to RGB")
            image = image.convert("RGB")
        
        print(f"Image size: {image.size}")
        
        # Resize image to a standard size
        image = image.resize((224, 224))
        print("Image resized to 224x224")
        
        # Convert to numpy array
        img_array = np.array(image)
        print(f"Converted to numpy array with shape {img_array.shape}")
        
        # Convert to PyTorch tensor using torch.tensor() instead of from_numpy
        print("Converting to PyTorch tensor using torch.tensor()")
        img_tensor = torch.tensor(img_array, dtype=torch.float32) / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # Change from HWC to CHW format
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        print(f"Converted to tensor with shape {img_tensor.shape}")
        
        # Normalize the tensor
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # Perform inference
        with torch.no_grad():
            outputs = model(img_tensor)
        
        # Get predictions
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0].tolist()
        
        # Create dictionary of class probabilities
        class_probs = {model.config.id2label[i]: float(prob) for i, prob in enumerate(probabilities)}
        print(f"Predictions: {class_probs}")
        
        return class_probs
    except Exception as e:
        print(f"Error in classification: {e}")
        import traceback
        traceback.print_exc()
        return {"angular_leaf_spot": 0.0, "bean_rust": 0.0, "healthy": 0.0}

# Load model
print("Loading model...")
model_name = "nateraw/vit-base-beans"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
print(f"Model loaded: {model_name}")
print(f"Model classes: {model.config.id2label}")

# Create Gradio interface
title = "Bean Disease Classification"
description = """
This application uses a Vision Transformer (ViT) model to classify bean leaves into three categories:
- angular_leaf_spot: Beans affected by Angular Leaf Spot disease
- bean_rust: Beans affected by Bean Rust disease
- healthy: Healthy bean leaves

Upload an image of bean leaves to get a classification.
"""

examples = [
    ["sample_images/angular_leaf_spot_1.jpg"],
    ["sample_images/bean_rust_1.jpg"],
    ["sample_images/healthy_1.jpg"]
]

# Create the interface
print("Creating Gradio interface...")
iface = gr.Interface(
    fn=classify_bean,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title=title,
    description=description,
    examples=examples,
    allow_flagging="never"
)

# Launch the app
if __name__ == "__main__":
    print("Launching Gradio interface...")
    iface.launch()