"""
NANO-042: Inline reasoning extractor for models that don't separate thinking.

When llama.cpp is configured with --reasoning-format none (or not configured),
models like Qwen3 dump <think>...</think> blocks inline in the content field.
This PostProcessor strips those blocks from the response and stashes the
reasoning in context.metadata["reasoning"] so that:
  1. HistoryRecorder can persist it to JSONL
  2. The pipeline result carries it to the GUI thought bubble
  3. TTS never speaks reasoning content

MUST be registered BEFORE HistoryRecorder in the pipeline.
"""

import re

from .base import PipelineContext, PostProcessor


# Regex to match <think>...</think> blocks, including across newlines
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class ReasoningExtractor(PostProcessor):
    """
    PostProcessor that extracts inline <think>...</think> blocks from LLM output.

    Only acts when the provider did NOT already separate reasoning
    (i.e., context.metadata["reasoning"] is empty/absent).
    """

    @property
    def name(self) -> str:
        return "reasoning_extractor"

    def process(self, context: PipelineContext, response: str) -> str:
        # If reasoning was already parsed by the provider, skip extraction
        if context.metadata.get("reasoning"):
            return response

        # Find all <think>...</think> blocks
        matches = _THINK_RE.findall(response)
        if not matches:
            return response

        # Combine all reasoning blocks
        reasoning = "\n".join(m.strip() for m in matches if m.strip())

        # Strip the <think> blocks from the response content
        cleaned = _THINK_RE.sub("", response).strip()

        # Stash reasoning in metadata for downstream consumers
        if reasoning:
            context.metadata["reasoning"] = reasoning

        return cleaned
