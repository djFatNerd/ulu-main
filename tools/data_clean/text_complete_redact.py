#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text-only building info completion & redaction via local HF LLM.
- Reads a FeatureCollection JSON with N features.
- For each feature: send properties (text-only) to an instruction model.
- The model outputs STRICT JSON following the required schema.
- image_loc (pixel-space localization) is passed through unchanged if present.
- Model auto-downloads from Hugging Face if missing.

Usage:
  python text_complete_redact.py \
    --in input.json \
    --out output.json \
    --model-id Qwen/Qwen2.5-7B-Instruct \
    --device auto \
    --max-new-tokens 512 \
    --temperature 0.2
"""

import argparse
import json
import sys
import re
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer
)

SYSTEM_PROMPT = """You are an expert data anonymizer and land-use annotator for urban planning.
Work ONLY from the given TEXT fields in the input. Do NOT use the internet or any outside knowledge.
Tasks:
1) Infer land-use labels (primary/secondary/tertiary) and coarse time (start_decade) ONLY from the text.
2) OUTPUT MUST BE STRICT JSON matching the schema below.
3) Absolutely DO NOT include any real-world identifying information in the output
   (no names, addresses, postcodes, cities, states, coordinates, OSM/provider IDs).
4) If evidence is insufficient, return null for that field and lower confidence.
5) In "provenance.evidence", include short quotes or paraphrases from the INPUT TEXT ONLY.
   Avoid proper nouns that could reveal the true location.
6) Do not output any field not in the schema.
7) Do not modify "image_loc"; it will be added by the caller from the input if present.
"""

USER_SCHEMA_HEADER = """Schema:
{
  "labels": {"primary":"string|null","secondary":"string|null","tertiary":"string|null"},
  "time": {"start_decade":"string|null"},
  "confidence": {"labels":0.0,"time":0.0},
  "provenance":[{"source":"text_only","evidence":["string","..."],"confidence":0.0}],
  "redactions":{"removed_fields":["string","..."]}
}

INPUT TEXT (verbatim JSON or key-value lines):
"""

REPAIR_INSTRUCTION = """Your previous output was not valid JSON. Return ONLY a valid JSON object that matches the schema.
Do not include any prose. Do not include code fences. JSON only."""

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input FeatureCollection JSON path")
    ap.add_argument("--out", dest="out_path", required=True, help="Output FeatureCollection JSON path")
    ap.add_argument("--model-id", dest="model_id", required=True,
                    help="HF model id, e.g. Qwen/Qwen2.5-7B-Instruct, mistralai/Mistral-7B-Instruct-v0.3, meta-llama/Llama-3.2-8B-Instruct")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device selection")
    ap.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens per sample")
    ap.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    ap.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    ap.add_argument("--batch-size", type=int, default=1, help="(Keep 1 for deterministic behavior)")
    return ap.parse_args()

def select_device_flag(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg

def load_model_and_tokenizer(model_id: str, device: str):
    # Auto-download if missing
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    if device == "cpu":
        model = model.to("cpu")
    return model, tok

def build_messages(raw_text_block: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_SCHEMA_HEADER + raw_text_block}
    ]

def model_chat_json(model, tokenizer, messages: List[Dict[str, str]],
                    max_new_tokens: int, temperature: float, top_p: float, device: str) -> str:
    """
    Use chat template if available; otherwise, simple concatenation.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()
    else:
        # Fallback for models without chat template
        sys_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        prompt = f"{sys_prompt}\n\n{user_prompt}\n\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()

def try_json_load(txt: str) -> Optional[Dict[str, Any]]:
    # First, direct parse
    try:
        return json.loads(txt)
    except Exception:
        pass
    # Try to extract the first {...} block
    m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
    if m:
        cand = m.group(0)
        try:
            return json.loads(cand)
        except Exception:
            return None
    return None

def repair_with_model(model, tokenizer, bad_text: str,
                      max_new_tokens: int, temperature: float, top_p: float, device: str) -> Optional[Dict[str, Any]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_SCHEMA_HEADER + "\n" + REPAIR_INSTRUCTION + "\n\nBAD OUTPUT:\n" + bad_text}
    ]
    fixed = model_chat_json(model, tokenizer, messages, max_new_tokens, temperature, top_p, device)
    return try_json_load(fixed)

def run_on_feature(model, tokenizer, feature: Dict[str, Any],
                   max_new_tokens: int, temperature: float, top_p: float, device: str) -> Dict[str, Any]:
    # Prepare raw text to feed the model: only textual attributes
    props = feature.get("properties", {}) or {}

    # image_loc is passed through unchanged (if present)
    image_loc = props.get("image_loc")

    # Build a compact text block: include full properties (the model must redact in output)
    # For safety, you may exclude geometry or coords, but we're following user instruction:
    # LLM handles redaction; we still advise not to include precise coordinates if you can avoid it.
    text_block = json.dumps(props, ensure_ascii=False, indent=2)

    messages = build_messages(text_block)
    raw_out = model_chat_json(model, tokenizer, messages, max_new_tokens, temperature, top_p, device)
    data = try_json_load(raw_out)

    if data is None:
        # One-shot repair
        data = repair_with_model(model, tokenizer, raw_out, max_new_tokens, temperature, top_p, device)
        if data is None:
            # Last resort: return a minimal null-filled structure
            data = {
                "labels": {"primary": None, "secondary": None, "tertiary": None},
                "time": {"start_decade": None},
                "confidence": {"labels": 0.0, "time": 0.0},
                "provenance": [{"source": "text_only", "evidence": [], "confidence": 0.0}],
                "redactions": {"removed_fields": []}
            }

    # Merge pass-through image_loc (if present)
    if image_loc is not None:
        data["image_loc"] = image_loc

    # Return a new Feature with same id/geometry but replaced properties
    new_feat = {
        "type": feature.get("type", "Feature"),
        "id": feature.get("id"),
        "properties": data
    }
    # Keep geometry untouched unless you want to drop it from public data
    if "geometry" in feature:
        new_feat["geometry"] = feature["geometry"]
    return new_feat

def main():
    args = parse_args()
    device = select_device_flag(args.device)

    # Load input
    with open(args.in_path, "r", encoding="utf-8") as f:
        fc = json.load(f)

    if not isinstance(fc, dict) or fc.get("type") != "FeatureCollection":
        print("ERROR: input must be a FeatureCollection JSON.", file=sys.stderr)
        sys.exit(1)

    features = fc.get("features", [])
    if not isinstance(features, list):
        print("ERROR: 'features' must be a list.", file=sys.stderr)
        sys.exit(1)

    # Load model & tokenizer (auto-download if missing)
    print(f"Loading model: {args.model_id} on {device} ...", file=sys.stderr)
    model, tokenizer = load_model_and_tokenizer(args.model_id, device)
    print("Model loaded.", file=sys.stderr)

    out_features = []
    for i, feat in enumerate(features):
        try:
            new_feat = run_on_feature(
                model, tokenizer, feat,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device
            )
            out_features.append(new_feat)
        except Exception as e:
            # On hard failure, push a null object but keep pipeline alive
            sys.stderr.write(f"[WARN] Feature {i} failed with error: {e}\n")
            fallback = {
                "type": feat.get("type", "Feature"),
                "id": feat.get("id"),
                "properties": {
                    "labels": {"primary": None, "secondary": None, "tertiary": None},
                    "time": {"start_decade": None},
                    "confidence": {"labels": 0.0, "time": 0.0},
                    "provenance": [{"source": "text_only", "evidence": [], "confidence": 0.0}],
                    "redactions": {"removed_fields": []}
                }
            }
            if "geometry" in feat:
                fallback["geometry"] = feat["geometry"]
            out_features.append(fallback)

        if (i + 1) % 20 == 0:
            print(f"Processed {i+1}/{len(features)} features...", file=sys.stderr)

    out_fc = {"type": "FeatureCollection", "features": out_features}
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(out_fc, f, ensure_ascii=False, indent=2)

    print(f"Done. Wrote {len(out_features)} features to {args.out_path}")

if __name__ == "__main__":
    main()
