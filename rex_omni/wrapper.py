#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main wrapper class for Rex Omni
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info, smart_resize

from .parser import convert_boxes_to_normalized_bins, parse_prediction
from .tasks import TASK_CONFIGS, TaskType, get_keypoint_config, get_task_config


class RexOmniWrapper:
    """
    High-level wrapper for Rex-Omni
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "transformers",
        system_prompt: str = "You are a helpful assistant",
        min_pixels: int = 16 * 28 * 28,
        max_pixels: int = 2560 * 28 * 28,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 0.8,
        top_k: int = 1,
        repetition_penalty: float = 1.05,
        skip_special_tokens: bool = False,
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the wrapper

        Args:
            model_path: Path to the model directory
            backend: Backend type ("transformers" or "vllm")
            system_prompt: System prompt for the model
            min_pixels: Minimum pixels for image processing
            max_pixels: Maximum pixels for image processing
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness in generation
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            skip_special_tokens: Whether to skip special tokens in output
            stop: Stop sequences for generation
            **kwargs: Additional arguments for model initialization
        """
        self.model_path = model_path
        self.backend = backend.lower()
        self.system_prompt = system_prompt
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # Store generation parameters
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.skip_special_tokens = skip_special_tokens
        self.stop = stop or ["<|im_end|>"]

        # Initialize model and processor
        self._initialize_model(**kwargs)

    def _initialize_model(self, **kwargs):
        """Initialize model and processor based on backend type"""
        print(f"Initializing {self.backend} backend...")

        if self.backend == "vllm":
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams

            # Initialize VLLM model
            self.model = LLM(
                model=self.model_path,
                tokenizer=self.model_path,
                tokenizer_mode=kwargs.get("tokenizer_mode", "slow"),
                limit_mm_per_prompt=kwargs.get(
                    "limit_mm_per_prompt", {"image": 10, "video": 10}
                ),
                max_model_len=kwargs.get("max_model_len", 4096),
                gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.8),
                tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
                trust_remote_code=kwargs.get("trust_remote_code", True),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k
                    not in [
                        "tokenizer_mode",
                        "limit_mm_per_prompt",
                        "max_model_len",
                        "gpu_memory_utilization",
                        "tensor_parallel_size",
                        "trust_remote_code",
                    ]
                },
            )

            # Initialize processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )

            # Set up sampling parameters
            self.sampling_params = SamplingParams(
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                top_k=self.top_k,
                temperature=self.temperature,
                skip_special_tokens=self.skip_special_tokens,
                stop=self.stop,
            )

            self.model_type = "vllm"

        elif self.backend == "transformers":
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            # Initialize transformers model
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
                attn_implementation=kwargs.get(
                    "attn_implementation", "flash_attention_2"
                ),
                device_map=kwargs.get("device_map", "auto"),
                trust_remote_code=kwargs.get("trust_remote_code", True),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k
                    not in [
                        "torch_dtype",
                        "attn_implementation",
                        "device_map",
                        "trust_remote_code",
                    ]
                },
            )

            # Initialize processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                use_fast=False,
            )

            self.model_type = "transformers"

        else:
            raise ValueError(
                f"Unsupported backend: {self.backend}. Choose 'transformers' or 'vllm'."
            )

    def inference(
        self,
        images: Union[Image.Image, List[Image.Image]],
        task: Union[str, TaskType],
        categories: Optional[Union[str, List[str]]] = None,
        keypoint_type: Optional[str] = None,
        visual_prompt_boxes: Optional[List[List[float]]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Perform inference on images for various vision tasks.

        Args:
            image: Input image in PIL.Image format.
            task: Task type. Available options:
                - "detection": Object detection with bounding boxes
                - "pointing": Point to objects with coordinates
                - "visual_prompting": Find similar objects based on reference boxes
                - "keypoint": Detect keypoints for persons/hands/animals
                - "ocr_box": Detect and recognize text in bounding boxes
                - "ocr_polygon": Detect and recognize text in polygons
                - "gui_grounding": Detect gui element and return in box format
                - "gui_pointing": Point to gui element and return in point format
            categories: Object categories to detect/locate. Required for most tasks.
                Examples: ["person", "car"], "dog", ["text"]
            keypoint_type: Type of keypoints for keypoint detection task.
                Options: "person", "hand", "animal"
            visual_prompt_boxes: Reference bounding boxes for visual prompting task.
                Format: [[x0, y0, x1, y1], ...] in absolute coordinates
            **kwargs: Additional arguments (reserved for future use)

        Returns:
            List of prediction dictionaries, one for each input image. Each dictionary contains:
            - success (bool): Whether inference succeeded
            - extracted_predictions (dict): Parsed predictions by category
            - raw_output (str): Raw model output text
            - inference_time (float): Total inference time in seconds
            - num_output_tokens (int): Number of generated tokens
            - num_prompt_tokens (int): Number of input tokens
            - tokens_per_second (float): Generation speed
            - image_size (tuple): Input image dimensions (width, height)
            - task (str): Task type used
            - prompt (str): Generated prompt sent to model

        Examples:
            # Object detection
            results = model.inference(
                images=image,
                task="detection",
                categories=["person", "car", "dog"]
            )

            # Point to specific objects
            results = model.inference(
                images=image,
                task="pointing",
                categories=["person"]
            )

            # Keypoint detection
            results = model.inference(
                images=image,
                task="keypoint",
                categories=["person"],
                keypoint_type="person"
            )

            # Visual prompting with reference boxes
            results = model.inference(
                images=image,
                task="visual_prompting",
                visual_prompt_boxes=[[100, 100, 200, 200]]
            )

            # OCR text detection
            results = model.inference(
                images=image,
                task="ocr_box",
                categories=["text"]
            )

            # Batch processing multiple images
            results = model.inference(
                images=[img1, img2, img3],
                task="detection",
                categories=["person", "car"]
            )
        """
        # Convert single image to list
        if isinstance(images, Image.Image):
            images = [images]

        # Convert task string to TaskType
        if isinstance(task, str):
            task = TaskType(task.lower())

        results = []

        for image in images:
            result = self._inference_single(
                image=image,
                task=task,
                categories=categories,
                keypoint_type=keypoint_type,
                visual_prompt_boxes=visual_prompt_boxes,
                **kwargs,
            )
            results.append(result)

        return results

    def _inference_single(
        self,
        image: Image.Image,
        task: TaskType,
        categories: Optional[Union[str, List[str]]] = None,
        keypoint_type: Optional[str] = None,
        visual_prompt_boxes: Optional[List[List[float]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform inference on a single image"""

        start_time = time.time()

        # Get image dimensions
        w, h = image.size

        # Generate prompt based on task
        final_prompt = self._generate_prompt(
            task=task,
            categories=categories,
            keypoint_type=keypoint_type,
            visual_prompt_boxes=visual_prompt_boxes,
            image_width=w,
            image_height=h,
        )

        # Calculate resized dimensions using smart_resize
        resized_height, resized_width = smart_resize(
            h,
            w,
            28,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        # Prepare messages
        if self.model_type == "transformers":
            # For transformers, use resized_height and resized_width
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "resized_height": resized_height,
                            "resized_width": resized_width,
                        },
                        {"type": "text", "text": final_prompt},
                    ],
                },
            ]
        else:
            # For VLLM, use min_pixels and max_pixels
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": self.min_pixels,
                            "max_pixels": self.max_pixels,
                        },
                        {"type": "text", "text": final_prompt},
                    ],
                },
            ]

        # Generate response
        if self.model_type == "vllm":
            raw_output, generation_info = self._generate_vllm(messages)
        else:
            raw_output, generation_info = self._generate_transformers(messages)

        # Parse predictions
        extracted_predictions = parse_prediction(
            text=raw_output,
            w=w,
            h=h,
            task_type=task.value,
        )

        # Calculate timing
        total_time = time.time() - start_time

        return {
            "success": True,
            "image_size": (w, h),
            "resized_size": (resized_width, resized_height),
            "task": task.value,
            "prompt": final_prompt,
            "raw_output": raw_output,
            "extracted_predictions": extracted_predictions,
            "inference_time": total_time,
            **generation_info,
        }

    def _generate_prompt(
        self,
        task: TaskType,
        categories: Optional[Union[str, List[str]]] = None,
        keypoint_type: Optional[str] = None,
        visual_prompt_boxes: Optional[List[List[float]]] = None,
        image_width: int = None,
        image_height: int = None,
    ) -> str:
        """Generate prompt based on task configuration"""

        task_config = get_task_config(task)

        if task == TaskType.VISUAL_PROMPTING:
            if visual_prompt_boxes is None:
                raise ValueError(
                    "Visual prompt boxes are required for visual prompting task"
                )

            # Convert boxes to normalized bins format
            word_mapped_boxes = convert_boxes_to_normalized_bins(
                visual_prompt_boxes, image_width, image_height
            )
            visual_prompt_dict = {"object_1": word_mapped_boxes}
            visual_prompt_json = json.dumps(visual_prompt_dict)

            return task_config.prompt_template.format(visual_prompt=visual_prompt_json)

        elif task == TaskType.KEYPOINT:
            if categories is None:
                raise ValueError("Categories are required for keypoint task")
            if keypoint_type is None:
                raise ValueError("Keypoint type is required for keypoint task")

            keypoints_list = get_keypoint_config(keypoint_type)
            if keypoints_list is None:
                raise ValueError(f"Unknown keypoint type: {keypoint_type}")

            keypoints_str = ", ".join(keypoints_list)
            categories_str = (
                ", ".join(categories) if isinstance(categories, list) else categories
            )

            return task_config.prompt_template.format(
                categories=categories_str, keypoints=keypoints_str
            )

        else:
            # Standard tasks (detection, pointing, OCR, etc.)
            if task_config.requires_categories and categories is None:
                raise ValueError(f"Categories are required for {task.value} task")

            if categories is not None:
                categories_str = (
                    ", ".join(categories)
                    if isinstance(categories, list)
                    else categories
                )
                return task_config.prompt_template.format(categories=categories_str)
            else:
                return task_config.prompt_template.format(categories="objects")

    def _generate_vllm(self, messages: List[Dict]) -> Tuple[str, Dict]:
        """Generate using VLLM model"""

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {"image": image_inputs}
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        # Generate
        generation_start = time.time()
        outputs = self.model.generate(
            [llm_inputs], sampling_params=self.sampling_params
        )
        generation_time = time.time() - generation_start

        generated_text = outputs[0].outputs[0].text

        # Extract token information
        output_tokens = outputs[0].outputs[0].token_ids
        num_output_tokens = len(output_tokens) if output_tokens else 0

        prompt_token_ids = outputs[0].prompt_token_ids
        num_prompt_tokens = len(prompt_token_ids) if prompt_token_ids else 0

        tokens_per_second = (
            num_output_tokens / generation_time if generation_time > 0 else 0
        )

        return generated_text, {
            "num_output_tokens": num_output_tokens,
            "num_prompt_tokens": num_prompt_tokens,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
        }

    def _generate_transformers(self, messages: List[Dict]) -> Tuple[str, Dict]:
        """Generate using Transformers model"""

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        generation_start = time.time()
        inputs = self.processor(
            text=[text],
            images=[messages[1]["content"][0]["image"]],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Prepare generation kwargs
        generation_kwargs = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.temperature > 0,  # Enable sampling if temperature > 0
            "pad_token_id": self.processor.tokenizer.eos_token_id,
        }

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        generation_time = time.time() - generation_start

        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=self.skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )[0]

        num_output_tokens = len(generated_ids_trimmed[0])
        num_prompt_tokens = len(inputs.input_ids[0])
        tokens_per_second = (
            num_output_tokens / generation_time if generation_time > 0 else 0
        )

        return output_text, {
            "num_output_tokens": num_output_tokens,
            "num_prompt_tokens": num_prompt_tokens,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
        }

    def get_supported_tasks(self) -> List[str]:
        """Get list of supported tasks"""
        return [task.value for task in TaskType]

    def get_task_info(self, task: Union[str, TaskType]) -> Dict[str, Any]:
        """Get information about a specific task"""
        if isinstance(task, str):
            task = TaskType(task.lower())

        config = get_task_config(task)
        return {
            "name": config.name,
            "description": config.description,
            "output_format": config.output_format,
            "requires_categories": config.requires_categories,
            "requires_visual_prompt": config.requires_visual_prompt,
            "requires_keypoint_type": config.requires_keypoint_type,
            "prompt_template": config.prompt_template,
        }
