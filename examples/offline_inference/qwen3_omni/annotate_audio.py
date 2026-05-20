# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Audio paralinguistic annotator — vllm-omni offline inference example.

Implements the full trans-qwen3-omni annotation pipeline as a self-contained
script that follows the same patterns as end2end.py.

Annotates audio files with:
  - ASR transcript + confidence
  - Free-text paralinguistic description (2-5 sentences)
  - 20-dimensional structured labels (300+ discrete labels)
  - Per-dimension confidence scores (0.0–1.0)

Usage:
    # Single file
    python annotate_audio.py --audio-path sample.wav --output results.jsonl

    # Directory (recursive)
    python annotate_audio.py --input-dir /data/audio --output results.jsonl

    # Resume an interrupted run
    python annotate_audio.py --input-dir /data/audio --output results.jsonl --resume

    # Custom model path + batch size
    python annotate_audio.py --input-dir /data/audio --output results.jsonl \\
        --model /path/to/Qwen3-Omni-30B-A3B-Instruct --batch-size 16

    # File list
    python annotate_audio.py --input-list files.txt --output results.jsonl
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from vllm import SamplingParams
from vllm.multimodal.media.audio import load_audio
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults
from vllm_omni.entrypoints.omni import Omni

SEED = 42
SAMPLING_RATE = 16000
SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}

# Talker codec EOS token ID (Qwen3-Omni specific)
TALKER_CODEC_EOS_TOKEN_ID = 2150


# ─── Label Taxonomy ──────────────────────────────────────────────────────────

TAXONOMY: dict[str, dict] = {
    "emotion_primary": {
        "description": "主要情感类别",
        "labels": [
            "neutral", "happy", "joyful", "excited", "amused", "satisfied",
            "proud", "loving", "grateful", "hopeful", "relieved",
            "sad", "depressed", "melancholic", "disappointed", "gloomy",
            "grieving", "lonely", "nostalgic",
            "angry", "furious", "irritated", "frustrated", "indignant",
            "resentful", "contemptuous",
            "fearful", "anxious", "nervous", "worried", "panicked", "horrified",
            "surprised", "astonished", "shocked", "confused",
            "disgusted", "bored", "tired", "embarrassed", "ashamed",
            "guilty", "jealous", "envious", "curious",
        ],
        "multi_select": True,
        "max_select": 3,
    },
    "emotion_mixed": {
        "description": "是否存在混合/矛盾情感",
        "labels": [
            "single_emotion", "mixed_emotions",
            "emotional_transition", "suppressed_emotion",
        ],
        "multi_select": False,
    },
    "emotion_intensity": {
        "description": "情感强度",
        "labels": [
            "intensity_none", "intensity_mild", "intensity_moderate",
            "intensity_strong", "intensity_extreme",
        ],
        "multi_select": False,
    },
    "speaking_style": {
        "description": "说话场景与风格",
        "labels": [
            "casual_conversation", "formal_speech", "intimate", "professional",
            "storytelling", "persuasive", "instructional", "news_reporting",
            "reading_aloud", "monologue", "dialogue", "debate", "interview",
            "commanding", "pleading", "consoling", "complaining", "mocking",
            "praising", "apologizing", "threatening", "negotiating", "joking",
            "dramatic", "childlike", "elderly_speech", "prayer_chant",
        ],
        "multi_select": True,
        "max_select": 3,
    },
    "attitude": {
        "description": "说话人态度",
        "labels": [
            "sincere", "sarcastic", "ironic", "polite", "rude",
            "assertive", "submissive", "dominant", "cooperative", "confrontational",
            "evasive", "direct", "indirect", "enthusiastic", "indifferent",
            "encouraging", "critical", "sympathetic", "empathetic", "doubtful",
        ],
        "multi_select": True,
        "max_select": 3,
    },
    "speech_rate": {
        "description": "语速",
        "labels": [
            "rate_very_fast", "rate_fast", "rate_normal", "rate_slow",
            "rate_very_slow", "rate_variable", "rate_accelerating", "rate_decelerating",
        ],
        "multi_select": False,
    },
    "volume": {
        "description": "音量",
        "labels": [
            "volume_very_loud", "volume_loud", "volume_normal", "volume_soft",
            "volume_very_soft", "volume_variable", "volume_fading", "volume_crescendo",
        ],
        "multi_select": False,
    },
    "pitch": {
        "description": "音调/音高",
        "labels": [
            "pitch_very_high", "pitch_high", "pitch_normal", "pitch_low",
            "pitch_very_low", "pitch_variable", "pitch_monotone",
            "pitch_rising", "pitch_falling", "pitch_rise_fall", "pitch_fall_rise",
        ],
        "multi_select": False,
    },
    "rhythm": {
        "description": "节奏韵律感",
        "labels": [
            "rhythm_steady", "rhythm_irregular", "rhythm_staccato",
            "rhythm_legato", "rhythm_rhythmic", "rhythm_choppy", "rhythm_flowing",
        ],
        "multi_select": False,
    },
    "paralinguistic_events": {
        "description": "非语言声音事件（可多选，标注出现过的类型）",
        "labels": [
            "laughter_full", "laughter_chuckle", "laughter_giggle",
            "laughter_snicker", "laughter_nervous", "laughter_forced",
            "crying_sobbing", "crying_whimpering", "crying_tearing_up", "crying_wailing",
            "breath_normal", "breath_heavy", "breath_gasp",
            "breath_sigh", "breath_deep_sigh", "breath_relief_sigh", "breath_frustrated_sigh",
            "breath_yawn",
            "throat_clearing", "cough", "cough_suppress", "sneeze", "hiccup",
            "lip_smack", "tongue_click", "teeth_chattering", "swallow",
            "sniffling", "nose_blowing",
        ],
        "multi_select": True,
        "max_select": 10,
    },
    "disfluency": {
        "description": "不流畅现象",
        "labels": [
            "fluent", "filler_uh_um", "filler_er", "filler_like", "filler_you_know",
            "word_repetition", "phrase_repetition", "false_start", "self_correction",
            "long_pause", "short_pause", "trailing_off", "rushed_speech",
            "incomplete_sentence", "word_search",
        ],
        "multi_select": True,
        "max_select": 6,
    },
    "voice_quality": {
        "description": "声音质量与嗓音特征",
        "labels": [
            "voice_clear", "voice_hoarse", "voice_breathy", "voice_nasal",
            "voice_whispery", "voice_creaky", "voice_shaky", "voice_strained",
            "voice_relaxed", "voice_resonant", "voice_thin", "voice_falsetto",
            "voice_fry", "voice_trembling", "voice_crying_tone", "voice_laughing_tone",
        ],
        "multi_select": True,
        "max_select": 4,
    },
    "speaker_gender": {
        "description": "说话人性别感知",
        "labels": ["gender_male", "gender_female", "gender_child", "gender_ambiguous"],
        "multi_select": False,
    },
    "speaker_age": {
        "description": "说话人年龄感知",
        "labels": [
            "age_child", "age_teenager", "age_young_adult",
            "age_middle_aged", "age_elderly", "age_ambiguous",
        ],
        "multi_select": False,
    },
    "language": {
        "description": "语言",
        "labels": [
            "mandarin_chinese", "english", "cantonese", "japanese", "korean",
            "french", "german", "spanish", "russian", "portuguese", "arabic",
            "mixed_language", "dialect", "other_language",
        ],
        "multi_select": True,
        "max_select": 3,
    },
    "accent": {
        "description": "口音特征",
        "labels": [
            "accent_standard", "accent_mild", "accent_strong",
            "accent_regional", "accent_foreign",
        ],
        "multi_select": False,
    },
    "noise_level": {
        "description": "背景噪声级别",
        "labels": [
            "noise_clean", "noise_slight", "noise_moderate", "noise_heavy",
            "noise_music_bg", "noise_crowd_bg", "noise_environment_bg",
        ],
        "multi_select": True,
        "max_select": 3,
    },
    "recording_condition": {
        "description": "录音环境条件",
        "labels": [
            "rec_studio", "rec_indoor_quiet", "rec_indoor_noisy", "rec_outdoor",
            "rec_telephone", "rec_broadcast", "rec_reverberant",
            "rec_close_mic", "rec_far_mic",
        ],
        "multi_select": False,
    },
    "audio_quality": {
        "description": "音频总体质量",
        "labels": [
            "quality_excellent", "quality_good", "quality_fair",
            "quality_poor", "quality_clipping", "quality_compressed",
        ],
        "multi_select": False,
    },
    "discourse_function": {
        "description": "话语功能/对话行为",
        "labels": [
            "statement", "question", "rhetorical_question", "exclamation",
            "command_request", "agreement", "disagreement", "backchannel",
            "turn_taking", "interruption", "topic_shift", "emphasis",
            "hedging", "self_disclosure", "humor_wit", "greeting_farewell",
        ],
        "multi_select": True,
        "max_select": 4,
    },
}


# ─── Prompt builder ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "你是一位专业的音频语言学标注专家，具备多语言语音识别（ASR）能力，"
    "能够精确转录音频内容并分析其中的副语言特征。"
    "你的任务是对输入音频同时完成语音转录和多维度副语言标注，输出结构化的JSON结果。"
)

_USER_PROMPT_TEMPLATE = """请对这段音频完成语音转录和完整的副语言分析，输出以下四个部分：

## 第一部分：语音转录（ASR）
逐字转录音频中所有说话人的语音内容，保留原始语言，不要翻译。
- 若有多位说话人，用 [说话人1]、[说话人2] 等前缀区分
- 保留明显的停顿（用 … 表示）和语气词（如"嗯""啊""呃"）
- 若音频为纯音乐/环境音/无人声，填写 "[无语音内容]"

## 第二部分：自由文本描述
用2-5句话自然描述这段音频的整体特征，包括说话人特征、情感状态、说话风格、语音质量和任何值得注意的副语言现象。

## 第三部分：结构化标签（多标签分类）
严格按照以下维度选择标签。每个维度只能从给定的标签列表中选择，不要创造新标签。

{taxonomy}

## 第四部分：置信度评分
对每个维度给出0.0~1.0的置信度分数，代表你对该维度判断的确定程度。
另外给出 ASR 转录的整体置信度（transcript_confidence）。

---

请用如下JSON格式输出（所有四个部分都在一个JSON对象中）：

```json
{{
  "transcript": "转录的语音文本...",
  "transcript_confidence": 0.95,
  "free_text_description": "你的自由文本描述...",
  "labels": {{
    "emotion_primary": ["label1", "label2"],
    "emotion_mixed": "label",
    "emotion_intensity": "label",
    "speaking_style": ["label1"],
    "attitude": ["label1", "label2"],
    "speech_rate": "label",
    "volume": "label",
    "pitch": "label",
    "rhythm": "label",
    "paralinguistic_events": ["label1", "label2"],
    "disfluency": ["label1"],
    "voice_quality": ["label1"],
    "speaker_gender": "label",
    "speaker_age": "label",
    "language": ["label1"],
    "accent": "label",
    "noise_level": ["label1"],
    "recording_condition": "label",
    "audio_quality": "label",
    "discourse_function": ["label1", "label2"]
  }},
  "confidence": {{
    "emotion_primary": 0.9,
    "emotion_mixed": 0.8,
    "emotion_intensity": 0.85,
    "speaking_style": 0.9,
    "attitude": 0.75,
    "speech_rate": 0.95,
    "volume": 0.95,
    "pitch": 0.9,
    "rhythm": 0.85,
    "paralinguistic_events": 0.9,
    "disfluency": 0.8,
    "voice_quality": 0.85,
    "speaker_gender": 0.95,
    "speaker_age": 0.7,
    "language": 0.99,
    "accent": 0.8,
    "noise_level": 0.9,
    "recording_condition": 0.85,
    "audio_quality": 0.95,
    "discourse_function": 0.8
  }}
}}
```

重要规则：
1. transcript 原文转录，不翻译，保留语气词和停顿
2. labels中每个字段的值必须来自上方列出的标签列表，不得使用列表外的标签
3. 单选维度输出字符串，多选维度输出字符串数组
4. confidence每项必须是0.0到1.0之间的浮点数
5. 若某特征在音频中完全无法判断，单选维度填 null，多选维度填 []，置信度填 0.0
6. 只输出JSON代码块，不要在JSON之外添加任何解释
"""


def _build_taxonomy_description() -> str:
    lines = []
    for dim, meta in TAXONOMY.items():
        multi = meta.get("multi_select", False)
        max_s = meta.get("max_select", 1)
        select_hint = f"(多选, 最多{max_s}个)" if multi else "(单选)"
        labels_str = ", ".join(meta["labels"])
        lines.append(f'- "{dim}" {select_hint} [{meta["description"]}]: {labels_str}')
    return "\n".join(lines)


_TAXONOMY_DESC = _build_taxonomy_description()
_USER_PROMPT = _USER_PROMPT_TEMPLATE.format(taxonomy=_TAXONOMY_DESC)

# Raw chat-template prompt string — follows the same pattern as end2end.py
_PROMPT_TEMPLATE = (
    f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
    f"{_USER_PROMPT}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


# ─── Output parsing ──────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Extract and parse the first JSON block from model output."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return json.loads(text[start:end])
    raise ValueError(f"No JSON found in model output:\n{text[:200]}")


def _validate_and_fix_labels(labels: dict) -> dict:
    clean = {}
    for dim, selected in labels.items():
        if dim not in TAXONOMY:
            continue
        valid = set(TAXONOMY[dim]["labels"])
        if isinstance(selected, list):
            clean[dim] = [lbl for lbl in selected if lbl in valid]
        elif selected is None:
            clean[dim] = None
        else:
            clean[dim] = selected if selected in valid else None
    return clean


def _make_error_result(audio_path: str, error: str, raw_output: str = "") -> dict:
    return {
        "audio_path": audio_path,
        "transcript": "",
        "transcript_confidence": 0.0,
        "free_text_description": "",
        "labels": {},
        "confidence": {},
        "parse_error": error,
        "raw_output": raw_output,
    }


def _parse_output(raw_text: str, audio_path: str) -> dict:
    try:
        parsed = _extract_json(raw_text)
        labels = _validate_and_fix_labels(parsed.get("labels", {}))
        confidence = {
            k: float(v)
            for k, v in parsed.get("confidence", {}).items()
            if k in TAXONOMY
        }
        return {
            "audio_path": audio_path,
            "transcript": parsed.get("transcript", ""),
            "transcript_confidence": float(parsed.get("transcript_confidence", 0.0)),
            "free_text_description": parsed.get("free_text_description", ""),
            "labels": labels,
            "confidence": confidence,
            "parse_error": None,
            "raw_output": raw_text,
        }
    except (json.JSONDecodeError, ValueError) as e:
        return _make_error_result(audio_path, str(e), raw_output=raw_text)


# ─── Core batch annotation ────────────────────────────────────────────────────

def annotate_batch(
    omni: Omni,
    audio_paths: list[str],
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    do_sample: bool = False,
) -> list[dict]:
    """
    Annotate a list of audio files in one vLLM inference pass.

    Follows the same generate() pattern as end2end.py:
    - Builds per-request prompt dicts with raw chat-template strings
    - Sets modalities=["text"] to run Thinker only (skips Talker + Code2Wav)
    - Uses py_generator=True for memory-efficient streaming over large batches
    - Maps vLLM sequential request IDs back to original audio_paths

    Returns a list of result dicts in the same order as audio_paths.
    """
    results: list[Optional[dict]] = [None] * len(audio_paths)
    prompts: list[dict] = []
    valid_indices: list[int] = []

    for idx, path in enumerate(audio_paths):
        try:
            audio_signal, sr = load_audio(str(path), sr=SAMPLING_RATE)
            prompts.append({
                "prompt": _PROMPT_TEMPLATE,
                "multi_modal_data": {"audio": (audio_signal.astype(np.float32), sr)},
                # Text-only output: skips Talker and Code2Wav stages
                "modalities": ["text"],
            })
            valid_indices.append(idx)
        except Exception as e:
            results[idx] = _make_error_result(str(path), f"LOAD_ERROR: {e}")

    if not prompts:
        return results

    thinker_params = SamplingParams(
        temperature=temperature if do_sample else 0.0,
        top_p=0.9,
        top_k=-1,
        max_tokens=max_new_tokens,
        repetition_penalty=1.05,
        seed=SEED,
    )
    # Talker / Code2Wav params (only active when num_stages > 1)
    talker_params = SamplingParams(
        temperature=0.9,
        top_k=50,
        max_tokens=4096,
        seed=SEED,
        detokenize=False,
        repetition_penalty=1.05,
        stop_token_ids=[TALKER_CODEC_EOS_TOKEN_ID],
    )
    code2wav_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=4096 * 16,
        seed=SEED,
        detokenize=True,
        repetition_penalty=1.1,
    )
    all_params = [thinker_params, talker_params, code2wav_params]
    sampling_params = all_params[: omni.num_stages]

    # Collect text outputs keyed by vLLM's sequential request_id ("0", "1", ...)
    raw_outputs: dict[str, str] = {}
    for stage_outputs in omni.generate(prompts, sampling_params, py_generator=True):
        output = stage_outputs.request_output
        if stage_outputs.final_output_type == "text":
            raw_outputs[output.request_id] = output.outputs[0].text

    for seq, idx in enumerate(valid_indices):
        path = audio_paths[idx]
        raw = raw_outputs.get(str(seq))
        if raw is not None:
            results[idx] = _parse_output(raw, str(path))
        else:
            results[idx] = _make_error_result(
                str(path), "NO_OUTPUT: vLLM returned no text for this request"
            )

    return results


# ─── File collection helpers ──────────────────────────────────────────────────

def collect_audio_files(
    input_dir: Optional[str],
    input_list: Optional[str],
    audio_path: Optional[str],
) -> list[Path]:
    if audio_path:
        return [Path(audio_path)]
    if input_dir:
        return sorted(
            f for f in Path(input_dir).rglob("*")
            if f.suffix.lower() in SUPPORTED_AUDIO_EXTS
        )
    if input_list:
        with open(input_list) as fh:
            return [Path(line.strip()) for line in fh if line.strip()]
    return []


def load_done_set(output_path: Path) -> set[str]:
    done: set[str] = set()
    if output_path.exists():
        with open(output_path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    done.add(rec["audio_path"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


# ─── Main ────────────────────────────────────────────────────────────────────

def main(args):
    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    files = collect_audio_files(args.input_dir, args.input_list, args.audio_path)
    if not files:
        print("No audio files found. Provide --audio-path, --input-dir, or --input-list.")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_set(output_path) if args.resume else set()
    pending = [f for f in files if str(f) not in done]

    print(f"Total files : {len(files)}")
    print(f"Already done: {len(done)}")
    print(f"To process  : {len(pending)}")
    print(f"Taxonomy    : {len(TAXONOMY)} dimensions, "
          f"{sum(len(v['labels']) for v in TAXONOMY.values())} labels")

    if not pending:
        print("Nothing to do.")
        return

    # Initialise Omni engine — same kwargs pattern as end2end.py
    omni = Omni(
        model=args.model,
        dtype=args.dtype,
        stage_configs_path=args.stage_configs_path,
        stage_init_timeout=args.stage_init_timeout,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
        log_stats=args.log_stats,
    )
    print(f"Engine ready. Stages: {omni.num_stages}")

    errors = 0
    t0 = time.time()
    batch_size = args.batch_size
    total_batches = (len(pending) + batch_size - 1) // batch_size
    processed = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for batch_num, batch_start in enumerate(range(0, len(pending), batch_size), start=1):
            batch = pending[batch_start: batch_start + batch_size]
            batch_paths = [str(f) for f in batch]

            try:
                results = annotate_batch(
                    omni,
                    batch_paths,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=args.do_sample,
                )
            except Exception as e:
                results = [
                    _make_error_result(p, f"BATCH_ERROR: {e}")
                    for p in batch_paths
                ]

            for result in results:
                if result.get("parse_error"):
                    errors += 1
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            processed += len(batch)
            if batch_num % args.log_every == 0 or batch_num == total_batches:
                elapsed = time.time() - t0
                avg = elapsed / processed
                eta = avg * (len(pending) - processed)
                print(
                    f"[batch {batch_num}/{total_batches}] "
                    f"[{processed}/{len(pending)} files] "
                    f"errors={errors}  elapsed={elapsed:.0f}s  "
                    f"avg={avg:.1f}s/file  eta={eta:.0f}s"
                )

    omni.close()
    print(f"\nDone. Results saved to {output_path}")
    print(f"Total errors: {errors}/{len(pending)}")


def parse_args():
    parser = FlexibleArgumentParser(
        description="Audio paralinguistic annotator using vllm-omni offline inference"
    )

    # Input sources (mutually exclusive)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--audio-path", "-a", type=str, default=None,
                     help="Path to a single audio file")
    src.add_argument("--input-dir", type=str, default=None,
                     help="Directory to scan for audio files recursively")
    src.add_argument("--input-list", type=str, default=None,
                     help="Text file with one audio path per line")

    # Output
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Skip files already present in the output JSONL")

    # Model
    parser.add_argument(
        "--model", type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="HF model name or local path",
    )
    parser.add_argument("--dtype", type=str, default="auto",
                        help="Model dtype (auto, half, float16, bfloat16)")

    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Max tokens for Thinker generation")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (used only when --do-sample is set)")
    parser.add_argument("--do-sample", action="store_true", default=False,
                        help="Enable sampling (default: greedy / temperature=0)")

    # Batching
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of audio files per vLLM batch")

    # Hardware
    parser.add_argument("--gpu", type=str, default=None,
                        help="CUDA device(s) to use, e.g. '1' or '0,1,2'")

    # vLLM-Omni engine parameters (same as end2end.py)
    parser.add_argument("--log-stats", action="store_true", default=False)
    parser.add_argument("--stage-init-timeout", type=int, default=300)
    parser.add_argument("--batch-timeout", type=int, default=5)
    parser.add_argument("--init-timeout", type=int, default=300)
    parser.add_argument("--shm-threshold-bytes", type=int, default=65536)
    parser.add_argument("--stage-configs-path", type=str, default=None,
                        help="Path to vllm-omni stage config YAML (optional)")

    # Logging
    parser.add_argument("--log-every", type=int, default=5,
                        help="Print progress every N batches")

    nullify_stage_engine_defaults(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
