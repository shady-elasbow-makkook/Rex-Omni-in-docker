
<div align=center>
  <img src="assets/logo.png" width=300 >
</div>

# Detect Anything via Next-Token Prediction

<div align=center>

<p align="center">
  <a href="https://rexthinker.github.io/">
    <img
      src="https://img.shields.io/badge/RexThinker-Website-Red?logo=afdian&logoColor=white&color=blue"
      alt="RexThinker Website"
    />
  </a>
  <a href="https://arxiv.org/abs/2506.04034">
    <img
      src="https://img.shields.io/badge/RexThinker-Paper-Red%25red?logo=arxiv&logoColor=red&color=yellow"
      alt="RexThinker Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/IDEA-Research/Rex-Thinker-GRPO-7B">
    <img 
        src="https://img.shields.io/badge/RexThinker-Weight-orange?logo=huggingface&logoColor=yellow" 
        alt="RexThinker weight on Hugging Face"
    />
  </a>
  <a href="https://huggingface.co/spaces/Mountchicken/Rex-Thinker-Demo">
    <img
      src="https://img.shields.io/badge/RexThinker-Demo-orange?logo=huggingface&logoColor=yellow" 
      alt="RexThinker Demo on Hugging Face"
    />
  </a>
  
</p>

</div>

> Rex-Omni is a 3B-parameter Multimodal Large Language Model (MLLM) that redefines object detection and a wide range of other visual perception tasks as a simple next-token prediction problem.

<p align="center"><img src="assets/teaser.png" width="95%"></p>



# Table of Contents


- [Detect Anything via Next-Token Prediction](#detect-anything-via-next-token-prediction)
- [Table of Contents](#table-of-contents)
  - [1. Installation ⛳️](#1-installation-️)
  - [2. Quick Start: Using Rex-Omni for Detection](#2-quick-start-using-rex-omni-for-detection)
      - [Initialization parameters (RexOmniWrapper)](#initialization-parameters-rexomniwrapper)
      - [Inference parameters (rex.inference)](#inference-parameters-rexinference)
  - [3. Cookbooks](#3-cookbooks)


## 1. Installation ⛳️

```bash
conda create -n rexomni -m python=3.10
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/IDEA-Research/Rex-Omni.git
cd Rex-Omni
pip install -v -e .
```

Test Installation
```bash
CUDA_VISIBLE_DEVICES=1 python tutorials/detection_example/detection_example.py
```

If the installation is successful, you will find a visualization of the detection results at `tutorials/detection_example/test_images/cafe_visualize.jpg`

## 2. Quick Start: Using Rex-Omni for Detection
Below is a minimal example showing how to run object detection using the `rex_omni` package.

```python
from PIL import Image
from rex_omni import RexOmniWrapper, RexOmniVisualize

# 1) Initialize the wrapper (model loads internally)
rex = RexOmniWrapper(
    model_path="IDEA-Research/Rex-Omni",   # HF repo or local path
    backend="transformers",                # or "vllm" for high-throughput inference
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
    "man", "woman", "yellow flower", "sofa", "robot-shope light",
    "blanket", "microwave", "laptop", "cup", "white chair", "lamp",
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
```

#### Initialization parameters (RexOmniWrapper)
- **model_path**: Hugging Face repo ID or a local checkpoint directory for the Rexe-Omni model.
- **backend**: "transformers" or "vllm".
  - **transformers**: easy to use, good baseline latency.
  - **vllm**: high-throughput, low-latency inference. Requires the `vllm` package and a compatible environment.
- **max_tokens**: Maximum number of tokens to generate for each output.
- **temperature**: Sampling temperature; higher values increase randomness (0.0 = deterministic/greedy).
- **top_p**: Nucleus sampling parameter; model samples from the smallest set of tokens with cumulative probability ≥ top_p.
- **top_k**: Top-k sampling; restricts sampling to the k most likely tokens.
- **repetition_penalty**: Penalizes repeated tokens; >1.0 discourages repetition.
- Optional advanced settings (supported via kwargs when constructing the wrapper):
  - Transformers: `torch_dtype`, `attn_implementation`, `device_map`, `trust_remote_code`, etc.
  - VLLM: `tokenizer_mode`, `limit_mm_per_prompt`, `max_model_len`, `gpu_memory_utilization`, `tensor_parallel_size`, `trust_remote_code`, etc.

#### Inference parameters (rex.inference)
- **images**: A single `PIL.Image.Image` or a list of images for batch inference.
- **task**: One of `"detection"`, `"pointing"`, `"visual_prompting"`, `"keypoint"`, `"ocr_box"`, `"ocr_polygon"`, `"gui_grounding"`, `"gui_pointing"`.
- **categories**: List of category names/phrases to detect or extract, e.g., `["person", "cup"]`. Used to build task prompts.
- **keypoint_type": Type of keypoints for keypoint detection task. Options: "person", "hand", "animal"
- **visual_prompt_boxes**: Reference bounding boxes for visual prompting task. Format: [[x0, y0, x1, y1], ...] in absolute coordinates

Returns a list of dictionaries (one per input image). Each dictionary includes:
- **raw_output**: The raw text generated by the LLM.
- **extracted_predictions**: Structured predictions parsed from the raw output, grouped by category.
  - For detection: `{category: [{"type": "box", "coords": [x0,y0,x1,y1]}, ...], ...}`
  - For pointing:  `{category: [{"type": "point", "coords": [x0,y0]}, ...], ...}`
  - For polygon: `{category: [{"type": "polygon", "coords": [x0,y0, ...]}, ...], ...}`
  - For keypointing: Structured Json

Tips:
- For best performance with VLLM, set `backend="vllm"` and tune `gpu_memory_utilization` and `tensor_parallel_size` according to your GPUs.

## 3. Cookbooks

We provide comprehensive tutorials for each supported task. Each tutorial includes both standalone Python scripts and interactive Jupyter notebooks.

|       Task       |                                                                Applications                                                               |   Demo |                  Python Example                    |                     Notebook                     |
|:----------------:|:-----------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------:|:------------------------------------------------:|:------------------------------------------------:|
| Detection |                               `object detection`                                 | ![img](assets/cafe_visualize.jpg)  | [code](tutorials/detection_example/detection_example.py)   | [notebook](tutorials/detection_example/_full_notebook.ipynb) |
|                  |                         `object referring`                      | ![img](assets/boys_visualize.jpg)   | [code](tutorials/detection_example/referring_example.py)      |                                                  |
|                  | `gui grounding` | ![img](assets/gui_visualize.jpg) | [code](tutorials/detection_example/gui_grounding_example.py)  |                                                  |
|                  |                    `layout grounding`    |  ![img](assets/layout_visualize.jpg)        | [code](tutorials/detection_example/layout_grouding_examle.py) |                                                  |
| Pointing |                               `object pointing`            |       ![img](assets/object_pointing_visualize.jpg)           |   [code](tutorials/pointing_example/object_pointing_example.py)   | [notebook](tutorials/pointing_example/_full_notebook.ipynb) |
|                  |                         `gui pointing`    |      ![img](assets/gui_pointing_visualize.jpg)              | [code](tutorials/pointing_example/gui_pointing_example.py)      |                                                  |
|                  | `affordance pointing` | ![img](assets/affordance_pointing_visualize.jpg) | [code](tutorials/pointing_example/affordance_pointing_example.py)  |                                                  |
| Visual prompting | `visual prompting` | ![img](assets/pigeons_visualize.jpg) | [code](tutorials/visual_prompting_example/visual_prompt_example.py) | [notebook](tutorials/visual_prompting_example/_full_tutorial.ipynb) |
| OCR | `ocr word box` | ![img](assets/ocr_word_box_visualize.jpg) | [code](tutorials/ocr_example/ocr_word_box_example.py) | [notebook](tutorials/ocr_example/_full_tutorial.ipynb) |
|                  | `ocr textline box` | ![img](assets/ocr_textline_box_visualize.jpg) | [code](tutorials/ocr_example/ocr_textline_box_example.py) | |
|                  | `ocr polygon` | ![img](assets/ocr_polygon_visualize.jpg) | [code](tutorials/ocr_example/ocr_polygon_example.py) |  |
| Keypointing | `person keypointing` | ![img](assets/person_keypointing_visualize.jpg) | [code](tutorials/keypointing_example/person_keypointing_example.py) | [notebook](tutorials/keypointing_example/_full_tutorial.ipynb)|
|             | `animal keypointing`   |     ![img](assets/animal_keypointing_visualize.jpg)                     |  [code](tutorials/keypointing_example/animal_keypointing_example.py)                                                |                                                  |