#!/usr/bin/env python3
"""
Bean Classification using Vision Transformer (ViT) model from Hugging Face.
"""

import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification, ViTImageProcessor
import numpy as np

def load_model():
    """Load the ViT model and processor from Hugging Face."""
    print("Loading model...")
    model_name = "nateraw/vit-base-beans"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    return model, processor

def preprocess_image(image_path, processor):
    """Preprocess the image for the model."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return image, inputs

def classify_image(model, inputs):
    """Classify the image using the model."""
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
    # Get class label and confidence
    predicted_class = model.config.id2label[predicted_class_idx]
    confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_idx].item()
    
    # Get all class probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].tolist()
    class_probs = {model.config.id2label[i]: prob for i, prob in enumerate(probabilities)}
    
    return predicted_class, confidence, class_probs

def display_results(image, predicted_class, confidence, class_probs):
    """Display the classification results."""
    print(f"\nPredicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    print("\nClass probabilities:")
    for class_name, prob in class_probs.items():
        print(f"  {class_name}: {prob:.2%}")
    
    # Display the image with prediction
    plt.figure(figsize=(8, 6))
    plt.imshow(np.array(image))
    plt.title(f"Predicted: {predicted_class} ({confidence:.2%})")
    plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Classify bean images using ViT model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the bean image")
    args = parser.parse_args()
    
    # Load model
    model, processor = load_model()
    
    # Preprocess image
    image, inputs = preprocess_image(args.image_path, processor)
    
    # Classify image
    predicted_class, confidence, class_probs = classify_image(model, inputs)
    
    # Display results
    display_results(image, predicted_class, confidence, class_probs)

if __name__ == "__main__":
    main()