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

    def __init__(self, model_path="HuggingFaceTB/SmolVLM-Instruct", processor_path=None, **kwargs):
        if not processor_path:
            processor_path = model_path
        from transformers import AutoProcessor, Idefics3ForConditionalGeneration

        assert osp.exists(model_path) or splitlen(model_path) == 2

        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.model = Idefics3ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float32, device_map="cuda"
        )
        # Video parameters with defaults
        self.nframe = kwargs.get("nframe", 25)
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
            "Video-MME"
        ]:
            print("Selected dataset")
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

    # def build_prompt_video_withtype(self, message, dataset, add_timestamps=False):
    #     """Build prompt with optional timestamp ranges"""
    #     from transformers.image_utils import load_image
    #     print(f"Message {message} Dataset {dataset}")
    #     prompt_parts = ["User:"]
    #     images = []
    #     timestamps = []

    #     for msg in message:
    #         if msg["type"] == "video":
    #             raise NotImplementedError("Not supported for now")

    #         elif msg["type"] == "image":
    #             prompt_parts.append("<image>")
    #             images.append(self.read_image(msg["value"]))

    #         elif msg["type"] == "text":
    #             prompt_parts.append(msg["value"].strip())

    #     if len(images) > self.nframe:
    #         print(f"We have {len(images)} images and we need to fit it into {self.nframe} frames.")

    #         # Select frame indices to reduce the images to self.nframe
    #         frame_indices = np.linspace(0, len(images) - 1, self.nframe, dtype=int).tolist()

    #         # Store the selected frames back in the 'images' list
    #         images = [images[i] for i in frame_indices]

    #         # Create a list of timestamps in mm:ss format
    #         timestamps = [f"{i // 60:02}:{i % 60:02}" for i in frame_indices]

    #         print("Selected frames:")
    #         for img, timestamp in zip(images, timestamps):
    #             print(f"Frame: {img['value']}, Timestamp: {timestamp}")

    #     elif len(images) <= self.nframe and len(images) != 0 :
    #         # If no reduction is needed, generate timestamps for all images
    #         timestamps = [f"{i // 60:02}:{i % 60:02}" for i in range(len(images))]

    #         print("No reduction needed. Frames and timestamps:")
    #         for img, timestamp in zip(images, timestamps):
    #             print(f"Frame: {img['value']}, Timestamp: {timestamp}")

    #     prompt = " ".join(prompt_parts)

    #     # Format based on dataset type
    #     if dataset in ["MLVU_MCQ", "MLVU_OpenEnded"]:
    #         prompt = prompt.replace("Options:", "Choices:")
    #         prompt = prompt.replace(
    #             "Please select the correct answer from the options above.",
    #             "Answer with the letter.",
    #         )
    #     elif dataset in [
    #         "TempCompass_MCQ",
    #         "TempCompass_Captioning",
    #         "TempCompass_YorN",
    #     ]:
    #         if dataset == "TempCompass_YorN":
    #             prompt += "\nAnswer yes or no."
    #         elif dataset == "TempCompass_MCQ":
    #             prompt = prompt.replace("Options:", "Choices:")
    #             prompt = prompt.replace(
    #                 "Please select the correct answer from the options above.",
    #                 "Answer with the letter.",
    #             )
    #     elif dataset in ["MVBench", "MVBench_MP4"]:
    #         if "Options:" in prompt:
    #             prompt = prompt.replace("Options:", "Choices:")
    #             prompt = prompt.replace(
    #                 "Please select the correct answer from the options above.",
    #                 "Answer with the letter.",
    #             )

    #     prompt += "<end_of_utterance>\nAssistant:"
    #     return prompt, images

    def build_prompt_video_withtype(self, message, dataset, add_timestamps=False):
        """Build prompt with optional timestamp ranges and handle trimming of image blocks"""
        from transformers.image_utils import load_image

        # print(f"Message {message} Dataset {dataset}")

        prompt_parts = ["User:"]
        processed_message = []
        image_blocks = []
        images = []
        timestamps = []
        nframe = 25  # TODO: check why self.nframe is None at this point of the code
        # print(f"Num frames max {nframe}")
        # Group consecutive image blocks
        current_block = []
        for msg in message:
            if msg["type"] == "image":
                current_block.append(msg)
            else:
                # If we encounter a non-image message and the current block is not empty
                if current_block:
                    image_blocks.append(current_block)  # Store the current block
                    current_block = []  # Reset for the next block
                processed_message.append(
                    msg
                )  # Add the non-image message directly to the processed message
        if current_block:
            image_blocks.append(current_block)  # Add the last block if it exists

        # Trim each image block if necessary
        for block in image_blocks:
            if len(block) > nframe:
                print(f"Trimming block of {len(block)} images to {nframe} frames.")
                frame_indices = np.linspace(
                    0, len(block) - 1, nframe, dtype=int
                ).tolist()
                trimmed_block = [block[i] for i in frame_indices]

                # Generate timestamps for the trimmed block
                block_timestamps = [f"{i // 60:02}:{i % 60:02}" for i in frame_indices]
            else:
                trimmed_block = block
                block_timestamps = [
                    f"{i // 60:02}:{i % 60:02}" for i in range(len(block))
                ]

            images.extend(trimmed_block)
            timestamps.extend(block_timestamps)

            # Add the trimmed block to the processed message
            for img, ts in zip(trimmed_block, block_timestamps):
                ts_str = f"{ts}" if add_timestamps else ""
                processed_message.append(
                    {"type": "text", "value": f"Frame from {ts_str}:"}
                )
                processed_message.append(img)

        images = []
        # Rebuild the prompt
        for msg in processed_message:
            if msg["type"] == "image":
                prompt_parts.append("<image>")
                images.append(self.read_image(msg["value"]))  # Load the image
            elif msg["type"] == "text":
                prompt_parts.append(msg["value"].strip())

        # Combine prompt parts
        prompt = " ".join(prompt_parts)

        # print(prompt_parts)

        # Format prompt based on dataset type
        if dataset in ["MLVU_MCQ", "MLVU_OpenEnded"]:
            prompt = prompt.replace("Options:", "Choices:")
            prompt = prompt.replace(
                "Please select the correct answer from the options above.",
                "Answer with the letter.",
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
                prompt = prompt.replace("Options:", "Choices:")
                prompt = prompt.replace(
                    "Please select the correct answer from the options above.",
                    "Answer with the letter.",
                )
        elif dataset in ["Video-MME"]:
            if "Options:" in prompt:
                prompt = prompt.replace("Options:", "Choices:")
                prompt = prompt.replace(
                    "Please select the correct answer from the options above.",
                    "Answer with the letter.",
                )
        else:
            raise NotImplementedError(f"{dataset} not found")

        prompt += "<end_of_utterance>\nAssistant:"
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
