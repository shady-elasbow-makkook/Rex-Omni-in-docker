from PIL import Image

from rex_omni import RexOmniVisualize, RexOmniWrapper

# 1) Initialize the wrapper (model loads internally)
rex = RexOmniWrapper(
    model_path="IDEA-Research/Rex-Omni",  # HF repo or local path
    backend="transformers",  # or "vllm" for high-throughput inference
    # Inference/generation controls (applied across backends)
    max_tokens=2048,
    temperature=0.0,
    top_p=0.05,
    top_k=1,
    repetition_penalty=1.05,
)

# 2) Prepare input
image = Image.open("tutorials/detection_example/test_images/cafe.jpg").convert("RGB")
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

# 3) Run detection
results = rex.inference(images=image, task="detection", categories=categories)
result = results[0]

# 4) Visualize
vis = RexOmniVisualize(
    image=image,
    predictions=result["extracted_predictions"],
    font_size=20,
    draw_width=5,
    show_labels=True,
)
vis.save("tutorials/detection_example/test_images/cafe_visualize.jpg")
