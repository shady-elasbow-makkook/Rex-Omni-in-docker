# Rex-Omni Tutorials Overview

## Quick Reference Table

| Task | Python Script | Description | Notebook |
|------|---------------|-------------|----------|
| **Detection** | | | |
| | `detection_example.py` | Basic object detection with multiple categories | `_full_notebook.ipynb` |
| | `referring_example.py` | Referring expression comprehension | `_full_notebook.ipynb` |
| | `layout_grouding_examle.py` | Document layout grounding | `_full_notebook.ipynb` |
| | `gui_grounding_example.py` | GUI element detection | `_full_notebook.ipynb` |
| **OCR** | | | |
| | `ocr_word_box_example.py` | Word-level OCR (box format) | `_full_tutorial.ipynb` |
| | `ocr_textline_box_example.py` | Text line OCR (box format) | `_full_tutorial.ipynb` |
| | `ocr_polygon_example.py` | Text line OCR (polygon format) | `_full_tutorial.ipynb` |
| **Pointing** | | | |
| | `object_pointing_example.py` | Point to specific objects | `_full_tutorial.ipynb` |
| | `affordance_pointing_example.py` | Point to functional parts | `_full_tutorial.ipynb` |
| | `gui_pointing_example.py` | Point to GUI elements | `_full_tutorial.ipynb` |
| **Keypointing** | | | |
| | `person_keypointing_example.py` | Human pose keypoint detection | `_full_tutorial.ipynb` |
| | `animal_keypointing_example.py` | Animal keypoint detection | `_full_tutorial.ipynb` |
| **Visual Prompting** | | | |
| | `visual_prompt_example.py` | Visual prompting with box guidance | `_full_tutorial.ipynb` |

## Usage

### Run Python Scripts
```bash
python tutorials/[task]/[script_name].py
```

### Open Jupyter Notebooks
```bash
jupyter notebook tutorials/[task]/_full_tutorial.ipynb
```

## Directory Structure
```
tutorials/
├── detection_example/
│   ├── detection_example.py
│   ├── referring_example.py
│   ├── layout_grouding_examle.py
│   ├── gui_grounding_example.py
│   └── _full_notebook.ipynb
├── ocr_example/
│   ├── ocr_word_box_example.py
│   ├── ocr_textline_box_example.py
│   ├── ocr_polygon_example.py
│   └── _full_tutorial.ipynb
├── pointing_example/
│   ├── object_pointing_example.py
│   ├── affordance_pointing_example.py
│   ├── gui_pointing_example.py
│   └── _full_tutorial.ipynb
├── keypointing_example/
│   ├── person_keypointing_example.py
│   ├── animal_keypointing_example.py
│   └── _full_tutorial.ipynb
└── visual_prompting_example/
    ├── visual_prompt_example.py
    └── _full_tutorial.ipynb
```
