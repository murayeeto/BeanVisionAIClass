#!/usr/bin/env python3
"""
Script to create synthetic sample bean images for testing.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_sample_images():
    """Create synthetic bean leaf images for testing."""
    print("Creating sample bean images...")
    
    # Create sample_images directory if it doesn't exist
    os.makedirs("sample_images", exist_ok=True)
    
    # Create sample images for each class
    create_angular_leaf_spot_images()
    create_bean_rust_images()
    create_healthy_bean_images()
    
    print("Sample images created successfully!")

def create_base_leaf(width=300, height=400, color=(100, 180, 100)):
    """Create a base leaf image."""
    # Create a base image with a light background
    image = Image.new('RGB', (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(image)
    
    # Draw a leaf shape
    leaf_points = [
        (width//2, 20),  # Top point
        (width-40, height//3),  # Right top
        (width-20, height//2),  # Right middle
        (width-60, height-50),  # Right bottom
        (width//2, height-20),  # Bottom point
        (60, height-50),  # Left bottom
        (20, height//2),  # Left middle
        (40, height//3),  # Left top
    ]
    
    # Draw the leaf
    draw.polygon(leaf_points, fill=color)
    
    # Add a stem
    draw.rectangle([(width//2-5, height-20), (width//2+5, height)], fill=(80, 120, 80))
    
    # Add some veins
    for i in range(5):
        start_x = width//2
        start_y = height//4 + i * height//8
        end_x = width//2 + (i % 2) * width//4 - (not i % 2) * width//4
        end_y = start_y + height//10
        draw.line([(start_x, start_y), (end_x, end_y)], fill=(80, 150, 80), width=2)
    
    # Apply slight blur for realism
    image = image.filter(ImageFilter.GaussianBlur(radius=1))
    
    return image

def create_angular_leaf_spot_images():
    """Create images of beans with angular leaf spot disease."""
    for i in range(1, 3):
        # Create base leaf
        image = create_base_leaf(color=(90, 160, 90))
        draw = ImageDraw.Draw(image)
        
        # Add angular spots (characteristic of angular leaf spot)
        num_spots = np.random.randint(5, 15)
        for _ in range(num_spots):
            x = np.random.randint(50, 250)
            y = np.random.randint(50, 350)
            size = np.random.randint(10, 30)
            
            # Angular spots have more defined edges
            points = [
                (x, y),
                (x + size, y + np.random.randint(-5, 5)),
                (x + np.random.randint(-5, 5), y + size),
                (x - size//2, y + np.random.randint(-5, 5))
            ]
            
            draw.polygon(points, fill=(150, 120, 60))
        
        # Save the image
        filename = f"sample_images/angular_leaf_spot_{i}.jpg"
        image.save(filename)
        print(f"Created {filename}")

def create_bean_rust_images():
    """Create images of beans with bean rust disease."""
    for i in range(1, 3):
        # Create base leaf
        image = create_base_leaf(color=(110, 170, 90))
        draw = ImageDraw.Draw(image)
        
        # Add rust spots (characteristic of bean rust)
        num_spots = np.random.randint(10, 25)
        for _ in range(num_spots):
            x = np.random.randint(50, 250)
            y = np.random.randint(50, 350)
            size = np.random.randint(5, 15)
            
            # Rust spots are more circular and reddish-brown
            draw.ellipse([(x-size, y-size), (x+size, y+size)], fill=(180, 80, 40))
        
        # Save the image
        filename = f"sample_images/bean_rust_{i}.jpg"
        image.save(filename)
        print(f"Created {filename}")

def create_healthy_bean_images():
    """Create images of healthy beans."""
    for i in range(1, 3):
        # Create base leaf with vibrant green (healthy)
        image = create_base_leaf(color=(80, 200, 80))
        
        # Add some natural variations for realism
        draw = ImageDraw.Draw(image)
        
        # Add a few small variations in color (but not disease spots)
        num_variations = np.random.randint(3, 8)
        for _ in range(num_variations):
            x = np.random.randint(50, 250)
            y = np.random.randint(50, 350)
            size = np.random.randint(3, 8)
            
            # Small natural variations
            draw.ellipse([(x-size, y-size), (x+size, y+size)],
                         fill=(70 + np.random.randint(0, 20),
                               190 + np.random.randint(0, 20),
                               70 + np.random.randint(0, 20)))
        
        # Save the image
        filename = f"sample_images/healthy_{i}.jpg"
        image.save(filename)
        print(f"Created {filename}")

if __name__ == "__main__":
    create_sample_images()