import torch
import os.path as osp
import warnings
from .base import BaseModel
from ..smp import splitlen
from PIL import Image
import numpy as np
import sys
import os
import math
from num2words import num2words
import datetime

sys.path.append("/fsx/miquel/smolvlmvideo")


class SmolVLMClip(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def sample_clip_indices(
        self,
        frames_per_clip: int,
        video_duration: float,
        sampling_fps: float,
        video_fps: float,
        max_clips: int,
    ):
        """
        Sample clips from video maintaining consecutive frames within each clip.
        """
        # Total number of frames in the video and effective clip duration
        tot_frames = int(video_duration * video_fps)
        clip_dur = frames_per_clip / sampling_fps

        # Determine number of possible clips, then cap at max_clips
        possible = math.ceil(video_duration / clip_dur)
        n_clips = min(max_clips, possible)

        # r > 0.5 means we can sample without repeating frames
        r = video_fps / sampling_fps
        clips, timestamps = [], []

        def pad(arr, target, pad_val):
            return (
                arr
                if arr.size >= target
                else np.concatenate(
                    (arr, np.full(target - arr.size, pad_val, dtype=np.int64))
                )
            )

        if r > 0.5:
            step = max(1, int(round(video_fps / sampling_fps)))
            seg_frames = int(round(frames_per_clip * step))
            valid = min(seg_frames, tot_frames)
            # If more clips than a single continuous segment, space them out
            gap = (
                ((tot_frames - seg_frames) // (n_clips - 1))
                if n_clips > 1 and tot_frames > seg_frames
                else 0
            )
            part = tot_frames // n_clips
            for i in range(n_clips):
                if part > seg_frames:
                    off = (part - seg_frames) // 2
                    idx = np.arange(off, off + seg_frames, step)
                    idx = np.clip(idx, 0, part - 1) + i * part
                else:
                    idx = np.arange(0, valid, step)
                    idx = pad(idx, frames_per_clip, valid - 1)
                    idx = np.clip(idx, 0, valid - 1) + i * gap
                idx = pad(idx, frames_per_clip, idx[-1] if idx.size else 0)
                clips.append(idx.astype(np.int64))
                timestamps.append((idx[0] / video_fps, idx[-1] / video_fps))
        else:
            reps = int(math.ceil(1 / r))
            seg_frames = int(round(frames_per_clip * r))
            base = np.repeat(np.arange(seg_frames), reps)
            gap = (
                ((tot_frames - seg_frames) // (n_clips - 1))
                if n_clips > 1 and tot_frames > seg_frames
                else 0
            )
            valid = min(seg_frames, tot_frames)
            base = pad(base, frames_per_clip, valid - 1)
            for i in range(n_clips):
                idx = np.clip(base, 0, valid - 1).astype(np.int64) + i * gap
                clips.append(idx)
                timestamps.append((idx[0] / video_fps, idx[-1] / video_fps))

        all_indices = np.concatenate(clips).tolist()
        return all_indices, timestamps

    def __init__(
        self,
        model_path="HuggingFaceTB/SmolVLM-2.2B-Instruct",
        checkpoint_path=None,
        sampling_frames=None,
        frames_per_clip=2,
        **kwargs,
    ):
        self.sampling_frames = sampling_frames
        self.frames_per_clip = frames_per_clip

        # Use SmolLMMProcessor for clip-based processing
        from transformers import AutoProcessor
        from smolvlm.model.processing_smollmm import SmolLMMProcessor

        assert osp.exists(model_path) or splitlen(model_path) == 2

        if checkpoint_path is None:
            checkpoint_path = model_path
        print(
            f"Checkpoint path set to {checkpoint_path}, Frame sampling to {sampling_frames}, "
            f"Frames per clip: {frames_per_clip}"
            f"Base model: {model_path}"
        )

        # Choose processor based on frames_per_clip
        if self.frames_per_clip > 1:
            self.processor = SmolLMMProcessor.from_pretrained(
                model_path,
                model_max_length=8192,
                padding_side="right",
            )
        else:
            raise ("This should be used only for frame averaging cases")

        from smolvlm.model.modeling_smollmm import SmolLMMForConditionalGeneration

        config = SmolLMMForConditionalGeneration.config_class.from_pretrained(
            checkpoint_path
        )
        config.frames_per_clip = self.frames_per_clip
        self.model = SmolLMMForConditionalGeneration.from_pretrained(
            checkpoint_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        # Video parameters
        self.fps = kwargs.get("fps", -1)
        if (
            model_path == "HuggingFaceTB/SmolVLM-Instruct"
            or model_path == "HuggingFaceTB/SmolVLM-2.2B-Instruct"
        ):
            self.resolution = 384
        elif (
            model_path == "HuggingFaceTB/SmolVLM-256M-Instruct"
            or model_path == "HuggingFaceTB/SmolVLM-500M-Instruct"
        ):
            self.resolution = 512
        else:
            raise (
                f"I don't recognize the model {model_path} and I cannot set the frame resolution"
            )
        print(f"Frame resolution set to {self.resolution}")
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
            # Configure processor for video input
            print(f"Double check: {self.resolution}")
            self.processor.image_processor.size = {"longest_edge": self.resolution}
            self.processor.image_processor.do_resize = True
            self.processor.image_processor.do_image_splitting = False
            formatted_messages, formatted_images = self.build_prompt_video_withtype(
                message, dataset, add_timestamps=add_timestamps
            )
        else:
            formatted_messages, formatted_images = self.build_prompt_default(message)

        # Process images and generate response
        images = (
            [formatted_images]
            if isinstance(formatted_images, Image.Image)
            else formatted_images
        )
        inputs = self.processor(
            text=formatted_messages,
            images=images,
            return_tensors="pt",
            padding=False,
            allow_mismatch=True,
        )

        # We can get the tokenized input after processor call
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1) :], skip_special_tokens=True
        )[0]

        return generated_text.strip()

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

    def build_prompt_video_withtype(self, message, dataset, add_timestamps=False):
        """Build prompt with clip-aware timestamp ranges"""

        prompt_parts = []
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

        # Add system message with proper format
        if system_message:
            prompt_parts.extend(
                ["System:", system_message["value"], "<end_of_utterance>\n"]
            )
        else:
            prompt_parts.extend(
                [
                    "System:",
                    "You are a helpful language and vision assistant. You are able to understand "
                    "the visual content that the user provides, "
                    "and assist the user with a variety of tasks using natural language.",
                    "<end_of_utterance>\n",
                ]
            )

        # Add User prefix and video intro
        # prompt_parts.extend(["User:", "Here are some clips sampled from a video:\n"])
        DEFAULT_VIDEO_INTRO = (
            "You are provided the following {frame_count} clips sampled"
            " from a {video_duration} [H:MM:SS] video. The clips:\n"
        )

        # Process image blocks with clip awareness
        text_messages = []
        current_block = []

        for msg in message:
            if msg["type"] == "image":
                current_block.append(msg)
            else:
                if current_block:
                    image_blocks.append(current_block)
                    current_block = []
                if msg.get("role") != "system":
                    text_messages.append(msg)

        if current_block:
            image_blocks.append(current_block)

        # Process image blocks with clip-based sampling
        for block in image_blocks:
            total_frames = len(block)
            block_duration = (
                total_frames / self.fps if self.fps > 0 else total_frames
            )  # Duration in seconds

            # Sample frames using the clip sampling function
            frame_indices, clip_times = self.sample_clip_indices(
                frames_per_clip=self.frames_per_clip,
                video_duration=block_duration,
                sampling_fps=1.0,  # Since we're working with frame indices directly
                video_fps=1.0,  # We're already in frame space
                max_clips=self.sampling_frames,
            )

            # Get frames based on computed indices
            trimmed_block = [block[i] for i in frame_indices]
            prompt_parts.extend(
                [
                    "User:",
                    DEFAULT_VIDEO_INTRO.format(
                        frame_count=num2words(len(clip_times)),
                        video_duration=str(datetime.timedelta(seconds=total_frames)),
                    ),
                ]
            )

            # Generate timestamps using clip timing information
            block_timestamps = []
            for start_time, end_time in clip_times:
                timestamp = (
                    f"{int(start_time // 60):02d}:{int(start_time % 60):02d} to "
                    f"{int(end_time // 60):02d}:{int(end_time % 60):02d}"
                )
                block_timestamps.append(timestamp)

            # Add frames and timestamps
            for i in range(0, len(trimmed_block), self.frames_per_clip):
                clip_frames = trimmed_block[i : i + self.frames_per_clip]
                ts = block_timestamps[i // self.frames_per_clip]
                ts_str = f"{ts}" if add_timestamps else ""

                prompt_parts.extend([f"Clip from {ts_str}:", "<image>"])
                for frame in clip_frames:
                    images.append(Image.open(frame["value"]).convert("RGB"))
            prompt_parts.append("\n")

        # Add remaining text and format
        for msg in text_messages:
            prompt_parts.append(msg["value"].strip())

        prompt_parts.extend(["<end_of_utterance>", "\nAssistant:"])
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
        print(f"{prompt} + Images: {len(images)}")
        return prompt, images
