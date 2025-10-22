#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic object detection example using Rex Omni
"""

import torch
from PIL import Image
from rex_omni import RexOmniVisualize, RexOmniWrapper


def main():
    # Model path - replace with your actual model path
    model_path = "IDEA-Research/Rex-Omni"

    # Create wrapper with custom parameters
    rex_model = RexOmniWrapper(
        model_path=model_path,
        backend="transformers",  # or "vllm" for faster inference
        max_tokens=2048,  # Reduced from 4096 as per docs recommendation
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
        torch_dtype=torch.float16,  # Use float16 instead of bfloat16 to save memory
        device_map="auto",  # Let it automatically distribute across available devices
    )

    # Load imag
    image_path = "tutorials/detection_example/test_images/cafe.jpg"  # Replace with your image path
    image = Image.open(image_path).convert("RGB")

    # Object detection
    categories = [
        "man",
        "woman",
        "yellow flower",
        "sofa",
        "robot-shope light",
        "blanket",
        "microwave",
        "laptop",
        "cup",
        "white chair",
        "lamp",
    ]

    results = rex_model.inference(images=image, task="detection", categories=categories)

    # Print results
    result = results[0]
    if result["success"]:
        predictions = result["extracted_predictions"]
        vis_image = RexOmniVisualize(
            image=image,
            predictions=predictions,
            font_size=20,
            draw_width=5,
            show_labels=True,
        )
        # Save visualization
        output_path = "tutorials/detection_example/test_images/cafe_visualize.jpg"
        vis_image.save(output_path)
        print(f"Visualization saved to: {output_path}")

    else:
        print(f"Inference failed: {result['error']}")


if __name__ == "__main__":
    main()
