# Bean Classification with Vision Transformer

This project demonstrates how to use the Vision Transformer (ViT) model from Hugging Face for classifying different types of beans.

## Model Information

The model used in this project is [nateraw/vit-base-beans](https://huggingface.co/nateraw/vit-base-beans), which is trained to classify beans into three categories:
- angular_leaf_spot
- bean_rust
- healthy

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

Run the classification script with an image path:

```
python classify_beans.py --image_path path/to/your/bean/image.jpg
```

### Web Interface

Run the web interface for interactive classification:

```
python app.py
```

Then open your browser and navigate to the URL shown in the terminal (typically http://127.0.0.1:7860).

## Project Structure

- `classify_beans.py`: Script for classifying bean images from the command line
- `app.py`: Web interface for interactive classification
- `requirements.txt`: Required Python dependencies
- `sample_images/`: Directory containing sample bean images for testing

## Sample Images

The sample_images directory contains example images of beans with different conditions that you can use to test the model.