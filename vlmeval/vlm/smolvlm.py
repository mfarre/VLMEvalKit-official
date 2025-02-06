import torch
import os.path as osp
import warnings
from .base import BaseModel
from ..smp import splitlen
from PIL import Image
import numpy as np

import os
import math


class SmolVLM(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(
        self,
        model_path="HuggingFaceTB/SmolVLM-Instruct",
        checkpoint_path=None,
        sampling_frames=None,
        **kwargs,
    ):
        self.sampling_frames = sampling_frames
        from transformers import AutoProcessor, Idefics3ForConditionalGeneration

        assert osp.exists(model_path) or splitlen(model_path) == 2

        if checkpoint_path is None:
            checkpoint_path = model_path
        print(
            f"Checkpoint path set to {checkpoint_path} and Frame sampling to {sampling_frames}"
        )

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Idefics3ForConditionalGeneration.from_pretrained(
            checkpoint_path, torch_dtype=torch.float32, device_map="cuda"
        )
        # Video parameters with defaults
        # self.nframe = kwargs.get("nframe", 25)
        self.fps = kwargs.get("fps", -1)  # Default to using nframe instead of fps
        self.resolution = 384

        kwargs_default = {"max_new_tokens": 512, "use_cache": True}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config."
        )
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None, add_timestamps=True):
        if dataset in [
            "MMBench_DEV_EN",
            "MMBench_TEST_EN",
            "MMBench_DEV_CN",
            "MMBench_TEST_CN",
            "MMBench",
            "MMBench_CN",
            "MMBench_DEV_EN_V11",
            "MMBench_DEV_CN_V11",
            "MMBench_TEST_EN_V11",
            "MMBench_TEST_CN_V11",
            "MMBench_V11",
            "MMBench_CN_V11",
            "CCBench",
        ]:
            formatted_messages, formatted_images = self.build_prompt_mmbench(message)
        elif dataset in ["MMMU_DEV_VAL", "MMMU_TEST"]:
            formatted_messages, formatted_images = self.build_prompt_mmmu(message)
        elif dataset in ["MathVista_MINI"]:
            formatted_messages, formatted_images = self.build_prompt_mathvista(message)
        elif dataset in [
            "MME",
            "MMVet",
            "OCRVQA_TEST",
            "OCRVQA_TESTCORE",
            "TextVQA_VAL",
            "ChartQA_TEST",
            "DocVQA_VAL",
            "DocVQA_TEST",
            "InfoVQA_VAL",
            "InfoVQA_TEST",
        ]:
            formatted_messages, formatted_images = self.build_prompt_default(
                message, add_brief=True
            )
        elif dataset == "HallusionBench":
            formatted_messages, formatted_images = self.build_prompt_default(
                message, add_yes_or_no=True
            )
        elif dataset in [
            "MMStar",
            "SEEDBench_IMG",
            "AI2D_TEST",
            "ScienceQA_VAL",
            "ScienceQA_TEST",
        ]:
            formatted_messages, formatted_images = self.build_prompt_puremcq(message)
        elif dataset in [
            "MLVU",
            "MLVU_MCQ",
            "MLVU_OpenEnded",
            "TempCompass",
            "TempCompass_MCQ",
            "TempCompass_Captioning",
            "TempCompass_YorN",
            "MVBench",
            "MVBench_MP4",
            "Video-MME",
        ]:
            # print("Selected dataset")
            # self.processor.image_processor.size = (384, 384)
            # self.processor.image_processor.do_resize = False
            self.processor.image_processor.size = {"longest_edge": 384}
            self.processor.image_processor.do_resize = True
            self.processor.image_processor.do_image_splitting = False
            formatted_messages, formatted_images = self.build_prompt_video_withtype(
                message, dataset, add_timestamps=add_timestamps
            )

        else:
            formatted_messages, formatted_images = self.build_prompt_default(message)

        images = (
            [formatted_images]
            if isinstance(formatted_images, Image.Image)
            else formatted_images
        )
        inputs = self.processor(
            text=formatted_messages, images=images, return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1) :], skip_special_tokens=True
        )[0]

        return generated_text.strip()

    def build_prompt_default(self, message, add_brief=False, add_yes_or_no=False):
        from transformers.image_utils import load_image

        prompt, images = "User:", []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
                prompt += "<image>"
            elif msg["type"] == "text":
                prompt += msg["value"].strip()
        if add_brief:
            prompt += "\nGive a very brief answer."
        if add_yes_or_no:
            prompt += "\nAnswer yes or no."
        prompt += "<end_of_utterance>\nAssistant:"
        return prompt, images

    def build_prompt_puremcq(self, message):
        from transformers.image_utils import load_image

        replace_mapping = {
            "\nOptions:": "\nChoices:",
            "Please select the correct answer from the options above.": "Answer with the letter.",
        }

        prompt, images = "User:", []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
                prompt += "<image>"
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction
        prompt += "<end_of_utterance>\nAssistant: Answer:"
        return prompt, images

    def build_prompt_mt(self, message):
        from transformers.image_utils import load_image

        prompt, images = "", []
        for msg in message:
            if msg["role"] == "user":
                prompt += "User: "
            elif msg["role"] == "assistant":
                prompt += "Assistant: "
            for item in msg["content"]:
                if item["type"] == "image":
                    img = load_image(item["value"])
                    images.append(img)
                elif item["type"] == "text":
                    prompt += item["value"].strip()
                prompt += "<end_of_utterance>\n"
        return prompt + "Assistant: "

    def build_prompt_mmbench(self, message):
        from transformers.image_utils import load_image

        replace_mapping = {
            "\nOptions:": "\nChoices:",
            "Please select the correct answer from the options above.": "Answer with a letter.",
        }

        prompt, images = "User:", []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
                prompt += "<image>"
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                # Swap hint and question
                if instruction.startswith("Hint:"):
                    hint, question = instruction.split("\nQuestion:")
                    question, choices = question.split("\nChoices:")
                    instruction = (
                        "Question:" + question + "\n" + hint + "\nChoices:" + choices
                    )
                prompt += instruction
        prompt += "<end_of_utterance>\nAssistant: Answer:"
        return prompt, images

    def build_prompt_mmmu(self, message):
        from transformers.image_utils import load_image

        replace_mapping = {
            "Question:": "",
            "Please select the correct answer from the options above.": "Answer with the letter.",
            "\nOptions:": "\nChoices:",
        }

        prompt, images, img_counter = "User: Question: ", [], 1
        for msg in message:
            if msg["type"] == "image":
                prompt += f"<image {img_counter}>:<image>\n"
                img_counter += 1
        img_counter = 1

        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
                prompt += f" <image {img_counter}> "
                img_counter += 1
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction.strip()
        prompt += "<end_of_utterance>\nAssistant:"
        if "A." in prompt and "B." in prompt:
            prompt += " Answer:"
        return prompt, images

    def build_prompt_mathvista(self, message):
        from transformers.image_utils import load_image

        replace_mapping = {
            "(A) ": "A. ",
            "(B) ": "B. ",
            "(C) ": "C. ",
            "(D) ": "D. ",
            "(E) ": "E. ",
            "(F) ": "F. ",
            "(G) ": "G. ",
            "(H) ": "H. ",
            "\nOptions:": "\nChoices:",
            "Hint: ": "",
        }

        prompt, images = "User:", []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
                prompt += "<image>"
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction.strip()

        prompt += "<end_of_utterance>\nAssistant:"
        if "A." in prompt and "B." in prompt:
            prompt += " Answer:"
        return prompt, images

    def chat_inner(self, message, dataset=None):
        formatted_messages, formatted_images = self.build_prompt_mt(message)
        images = (
            [formatted_images]
            if isinstance(formatted_images, Image.Image)
            else formatted_images
        )

        resulting_messages = [
            {
                "role": "user",
                "content": [{"type": "image"}]
                + [{"type": "text", "text": formatted_messages}],
            }
        ]
        prompt = self.processor.apply_chat_template(
            resulting_messages, add_generation_prompt=True
        )

        inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1) :], skip_special_tokens=True
        )[0]

        return generated_text.strip()

    def read_image(self, path):
        # Open image and convert to RGB
        # jpeg = Image.open(path).convert("RGB")
        # return self.resize_and_center_crop_pil(jpeg)
        return Image.open(path).convert("RGB")

    def build_prompt_video_withtype(self, message, dataset, add_timestamps=False):
        """Build prompt with optional timestamp ranges and handle trimming of image blocks"""
        from transformers.image_utils import load_image

        prompt_parts = []  # Remove "User:" since we'll add specific prefixes
        image_blocks = []
        images = []

        # Find system message first
        system_message = next(
            (
                msg
                for msg in message
                if msg["type"] == "text" and msg.get("role") == "system"
            ),
            None,
        )

        # Add system message with proper format if it exists
        if system_message:
            prompt_parts.extend(
                ["System:", system_message["value"], "<end_of_utterance>\n"]
            )
        else:
            # Adding default system message
            prompt_parts.extend(
                [
                    "System:",
                    "pay attention to the video and answer the question",
                    "<end_of_utterance>\n",
                ]
            )
        # Add User: prefix and "Here is the video:"
        prompt_parts.extend(["User:", "Here are some frames sampled from a video:\n"])

        # Process image blocks
        text_messages = []
        current_block = []

        for msg in message:
            if msg["type"] == "image":
                current_block.append(msg)
            else:
                if current_block:
                    image_blocks.append(current_block)
                    current_block = []
                if (
                    msg.get("role") != "system"
                ):  # Skip system message as it's already added
                    text_messages.append(msg)

        if current_block:
            image_blocks.append(current_block)

        # Process image blocks with trimming
        for block in image_blocks:
            if len(block) > self.sampling_frames:
                print(
                    f"\tTrimming block of {len(block)} images to {self.sampling_frames} frames."
                )
                frame_indices = np.linspace(
                    0, len(block) - 1, self.sampling_frames, dtype=int
                ).tolist()
                trimmed_block = [block[i] for i in frame_indices]
                block_timestamps = [f"{i // 60:02}:{i % 60:02}" for i in frame_indices]
            else:
                trimmed_block = block
                block_timestamps = [
                    f"{i // 60:02}:{i % 60:02}" for i in range(len(block))
                ]
            for img, ts in zip(trimmed_block, block_timestamps):
                ts_str = f"{ts}" if add_timestamps else ""
                prompt_parts.extend([f"Frame from {ts_str}:", "<image>"])
                images.append(self.read_image(img["value"]))
            prompt_parts.append("\n")

        # Add remaining text and final format
        for msg in text_messages:
            prompt_parts.append(msg["value"].strip())

        prompt_parts.append("<end_of_utterance>")
        prompt_parts.append("\nAssistant:")

        # Combine prompt parts
        prompt = " ".join(prompt_parts)

        # Format prompt based on dataset type
        if dataset in ["MLVU_MCQ", "MLVU_OpenEnded"]:
            prompt = prompt.replace(
                "Options:",
                "respond ONLY with one of the multiple choice letter options (A/B/C/D):",
            )

        elif dataset in [
            "TempCompass_MCQ",
            "TempCompass_Captioning",
            "TempCompass_YorN",
        ]:
            if dataset == "TempCompass_YorN":
                prompt += "\nAnswer yes or no."
            elif dataset == "TempCompass_MCQ":
                prompt = prompt.replace("Options:", "Choices:")
                prompt = prompt.replace(
                    "Please select the correct answer from the options above.",
                    "Answer with the letter.",
                )
        elif dataset in ["MVBench", "MVBench_MP4"]:
            if "Options:" in prompt:
                prompt = prompt.replace(
                    "Options:",
                    "respond ONLY with one of the multiple choice letter options (A/B/C/D):",
                )
                prompt = prompt.replace("Best option:(", "Answer:")
        elif dataset in ["Video-MME"]:
            if "Options:" in prompt:
                prompt = prompt.replace("Options:", "Choices:")
                prompt = prompt.replace(
                    "Please select the correct answer from the options above.",
                    "Answer with the letter.",
                )
        else:
            raise NotImplementedError(f"{dataset} not found")

        # prompt += "<end_of_utterance>\nAssistant:"
        # print(f"PROMPT {prompt}")
        print(f"{prompt} + Images: {len(images)}")
        return prompt, images

    def message_to_promptvideo(self, message):
        """Extract video path and question from message"""
        video_path = None
        question = None

        for msg in message:
            if msg["type"] == "video":
                video_path = msg["value"]
            elif msg["type"] == "text":
                question = msg["value"]

        return question, video_path
