"""
Stage 3c: Token-Gated LLM Report Generation (CP .tex core contribution).

Given:
  - top-k Evidence Tokens routed by Stage 3
  - sentence plan (anatomy keyword, topic)
  - trained W_proj (optional; falls back to identity)

Output:
  - LLM-generated report sentence conditioned on spatial token context

Two modes:
  1. text-only: format token metadata as structured text prompt (no W_proj needed)
  2. visual (VLM): extract CT voxel patches → send to Qwen2-VL / LLaVA (requires volume)

Usage (standalone generation):
    python -m ProveTok_Main_experiment.stage3c_generator \
        --trace_jsonl outputs/.../trace.jsonl \
        --tokens_pt  outputs/.../tokens.pt \
        --llm_backend ollama \
        --llm_model qwen2.5:7b
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .types import EvidenceToken, SentencePlan


@dataclass
class GeneratorConfig:
    # Backend: "ollama" | "openai" | "anthropic" | "huggingface"
    backend: str = "ollama"
    model: str = "qwen2.5:7b"
    ollama_host: str = "http://localhost:11434"
    timeout_s: float = 60.0
    temperature: float = 0.3
    max_tokens: int = 256
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    # HuggingFace backend options
    hf_device_map: str = "auto"
    hf_torch_dtype: str = "bfloat16"
    hf_token: Optional[str] = None
    # If True, include bbox coordinates in the prompt
    include_bbox: bool = True
    # If True, include split_score and level in the prompt
    include_scores: bool = True


_SYSTEM_PROMPT = (
    "You are an expert radiologist writing a CT report. "
    "You will be given evidence tokens extracted from a CT scan — each token "
    "describes a spatial region with its anatomical location and image features. "
    "Generate a single concise radiology report sentence that accurately describes "
    "the findings for the specified anatomy region. "
    "Write in standard radiology report style. Do not add disclaimers."
)


def _format_token_context(
    tokens: Sequence[EvidenceToken],
    anatomy_keyword: Optional[str],
    include_bbox: bool,
    include_scores: bool,
) -> str:
    """Format evidence tokens as a structured text block for the LLM prompt."""
    lines: List[str] = []
    if anatomy_keyword:
        lines.append(f"Target anatomy: {anatomy_keyword}")
    lines.append(f"Evidence tokens ({len(tokens)} total):")
    for i, tok in enumerate(tokens):
        parts = [f"  Token {i+1} (id={tok.token_id}, level={tok.level})"]
        if include_scores:
            parts.append(f"    split_score={tok.split_score:.3f}")
        if include_bbox:
            b = tok.bbox
            cx, cy, cz = b.center()
            parts.append(
                f"    center_xyz=({cx:.1f}, {cy:.1f}, {cz:.1f}), "
                f"vol={b.volume():.0f} voxels³"
            )
        lines.extend(parts)
    return "\n".join(lines)


def _build_generation_prompt(
    plan: SentencePlan,
    tokens: Sequence[EvidenceToken],
    include_bbox: bool,
    include_scores: bool,
) -> str:
    ctx = _format_token_context(tokens, plan.anatomy_keyword, include_bbox, include_scores)
    negation_note = " (negated finding expected)" if plan.is_negated else ""
    return (
        f"Topic sentence to generate{negation_note}: \"{plan.topic}\"\n\n"
        f"{ctx}\n\n"
        "Write one concise radiology report sentence for this finding:"
    )


@dataclass
class GeneratedSentence:
    sentence_index: int
    original_topic: str
    generated_text: str
    token_ids_used: List[int]
    error: Optional[str] = None


class Stage3cGenerator:
    """
    Stage 3c: LLM generates report sentence conditioned on evidence token context.

    This implements the CP .tex Token-Gated Generation concept:
    - Evidence tokens from Stage 2/3 become the "visual context"
    - LLM generates grounded report text from spatial token descriptions
    - With trained W_proj, token features are semantically aligned to text space
    """

    def __init__(self, cfg: GeneratorConfig) -> None:
        self.cfg = cfg
        self._client: Any = None
        self._hf_pipe: Any = None
        if cfg.backend == "openai":
            import openai  # type: ignore
            kw: Dict[str, Any] = {}
            if cfg.openai_api_key:
                kw["api_key"] = cfg.openai_api_key
            self._client = openai.OpenAI(**kw)
        elif cfg.backend == "anthropic":
            import anthropic  # type: ignore
            kw = {}
            if cfg.anthropic_api_key:
                kw["api_key"] = cfg.anthropic_api_key
            self._client = anthropic.Anthropic(**kw)
        elif cfg.backend == "huggingface":
            import os
            import torch  # type: ignore
            from transformers import pipeline  # type: ignore
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(cfg.hf_torch_dtype, torch.bfloat16)
            kw = dict(
                task="text-generation",
                model=cfg.model,
                torch_dtype=dtype,
                device_map=cfg.hf_device_map,
            )
            is_local = os.path.isdir(cfg.model)
            if not is_local and cfg.hf_token:
                kw["token"] = cfg.hf_token
            self._hf_pipe = pipeline(**kw)
            src = "local" if is_local else "HuggingFace Hub"
            print(f"[Stage 3c] Loaded model ({src}): {cfg.model}")

    # ------------------------------------------------------------------
    # LLM backend calls
    # ------------------------------------------------------------------

    def _call_ollama(self, user_prompt: str) -> str:
        import json as _json
        import urllib.request

        payload = _json.dumps(
            {
                "model": self.cfg.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": self.cfg.temperature,
                    "num_predict": self.cfg.max_tokens,
                },
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.cfg.ollama_host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            data = _json.loads(resp.read())
        return str(data.get("message", {}).get("content", ""))

    def _call_openai(self, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            timeout=self.cfg.timeout_s,
        )
        return str(resp.choices[0].message.content)

    def _call_anthropic(self, user_prompt: str) -> str:
        resp = self._client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return str(resp.content[0].text)

    def _call_huggingface(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        do_sample = self.cfg.temperature > 0.0
        outputs = self._hf_pipe(
            messages,
            max_new_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature if do_sample else None,
            do_sample=do_sample,
        )
        generated = outputs[0]["generated_text"]
        if isinstance(generated, list):
            return str(generated[-1].get("content", ""))
        return str(generated)

    def _call_llm(self, user_prompt: str) -> str:
        if self.cfg.backend == "ollama":
            return self._call_ollama(user_prompt)
        if self.cfg.backend == "openai":
            return self._call_openai(user_prompt)
        if self.cfg.backend == "anthropic":
            return self._call_anthropic(user_prompt)
        if self.cfg.backend == "huggingface":
            return self._call_huggingface(user_prompt)
        raise ValueError(f"Unknown backend: {self.cfg.backend!r}")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_sentence(
        self,
        plan: SentencePlan,
        cited_tokens: Sequence[EvidenceToken],
    ) -> GeneratedSentence:
        """Generate one report sentence conditioned on cited tokens."""
        prompt = _build_generation_prompt(
            plan,
            cited_tokens,
            self.cfg.include_bbox,
            self.cfg.include_scores,
        )
        try:
            text = self._call_llm(prompt).strip()
            # Take only the first sentence if LLM generates multiple
            first = text.split("\n")[0].strip()
            if first:
                text = first
            return GeneratedSentence(
                sentence_index=plan.sentence_index,
                original_topic=plan.topic,
                generated_text=text,
                token_ids_used=[t.token_id for t in cited_tokens],
            )
        except Exception as e:
            return GeneratedSentence(
                sentence_index=plan.sentence_index,
                original_topic=plan.topic,
                generated_text=plan.topic,  # fallback: return original topic
                token_ids_used=[t.token_id for t in cited_tokens],
                error=str(e),
            )

    def generate_report(
        self,
        plans: Sequence[SentencePlan],
        sentence_citations: Dict[int, List[int]],
        all_tokens: Sequence[EvidenceToken],
    ) -> List[GeneratedSentence]:
        """
        Generate all sentences for a report.

        Args:
            plans: sentence plans (from ReportSentencePlanner)
            sentence_citations: {sentence_index: [token_ids]} from Stage 3 routing
            all_tokens: full token bank for the case
        """
        token_map = {t.token_id: t for t in all_tokens}
        results: List[GeneratedSentence] = []
        for plan in plans:
            cited_ids = sentence_citations.get(plan.sentence_index, [])
            cited = [token_map[tid] for tid in cited_ids if tid in token_map]
            results.append(self.generate_sentence(plan, cited))
        return results
