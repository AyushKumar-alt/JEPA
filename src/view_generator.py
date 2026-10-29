"""
Context-Target View Generation for JEPA-inspired Representation Learning.
Constructs semantically meaningful context and target views from label-free state text.
Supports:
  1. 'span_mask' (preferred text JEPA / data2vec design: context is masked state, target is full state)
  2. 'sentence_split' (prefix vs suffix partition)
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import re
import random


@dataclass
class ContextTargetPair:
    state_id: str
    context_text: str
    target_text: str
    domain: str
    context_word_count: int
    target_word_count: int
    strategy: str


@dataclass
class ViewGenerationStats:
    total_states_evaluated: int
    valid_pairs_generated: int
    rejected_states_count: int
    rejection_reasons: Dict[str, int]
    avg_context_word_count: float
    avg_target_word_count: float
    identical_pairs_count: int
    overlapping_pairs_count: int
    empty_context_count: int
    empty_target_count: int


class ViewGenerator:
    """
    Generates aligned context and target views from state text.
    Ensures context and target are non-identical and informative.
    """

    def __init__(
        self,
        strategy: str = "span_mask",
        min_words_total: int = 8,
        min_context_words: int = 3,
        min_target_words: int = 3,
        context_ratio: float = 0.60,
        mask_ratio: float = 0.25,
        max_span_length: int = 4,
        seed: int = 42,
    ):
        self.strategy = strategy
        self.min_words_total = min_words_total
        self.min_context_words = min_context_words
        self.min_target_words = min_target_words
        self.context_ratio = context_ratio
        self.mask_ratio = mask_ratio
        self.max_span_length = max_span_length
        self.rng = random.Random(seed)

    def split_sentences(self, text: str) -> List[str]:
        """Splits text into sentences based on punctuation."""
        raw_sents = re.split(r"(?<=[.!?\n])\s+", text.strip())
        return [s.strip() for s in raw_sents if s.strip()]

    def _generate_span_mask_pair(
        self,
        state_text: str,
        state_id: str,
        domain: str,
    ) -> Optional[ContextTargetPair]:
        """
        Preferred data2vec/JEPA approach:
        Target: Complete uncorrupted state text.
        Context: State text with random spans replaced by [MASK].
        """
        words = state_text.strip().split()
        n = len(words)
        if n < self.min_words_total:
            return None

        target_masked_count = max(1, int(round(n * self.mask_ratio)))
        masked_indices = set()
        attempts = 0

        while len(masked_indices) < target_masked_count and attempts < 100:
            attempts += 1
            span_len = self.rng.randint(1, self.max_span_length)
            start = self.rng.randint(0, n - 1)
            for i in range(start, min(n, start + span_len)):
                masked_indices.add(i)
                if len(masked_indices) >= target_masked_count:
                    break

        masked_words = []
        i = 0
        while i < n:
            if i in masked_indices:
                masked_words.append("[MASK]")
                while i < n and i in masked_indices:
                    i += 1
            else:
                masked_words.append(words[i])
                i += 1

        context_text = " ".join(masked_words).strip()
        target_text = state_text.strip()

        if not context_text or not target_text or context_text == target_text:
            return None

        return ContextTargetPair(
            state_id=state_id,
            context_text=context_text,
            target_text=target_text,
            domain=domain,
            context_word_count=len(context_text.split()),
            target_word_count=len(target_text.split()),
            strategy="span_mask",
        )

    def _generate_split_pair(
        self,
        state_text: str,
        state_id: str,
        domain: str,
    ) -> Optional[ContextTargetPair]:
        """Prefix/suffix partition strategy."""
        words = state_text.strip().split()
        if len(words) < self.min_words_total:
            return None

        sentences = self.split_sentences(state_text)
        strategy = "sentence_split"

        if len(sentences) >= 2:
            num_sents = len(sentences)
            k = max(1, min(num_sents - 1, int(round(num_sents * self.context_ratio))))
            context_text = " ".join(sentences[:k]).strip()
            target_text = " ".join(sentences[k:]).strip()
        else:
            strategy = "word_span_split"
            num_words = len(words)
            m = max(self.min_context_words, min(num_words - self.min_target_words, int(round(num_words * self.context_ratio))))
            context_text = " ".join(words[:m]).strip()
            target_text = " ".join(words[m:]).strip()

        ctx_words = context_text.split()
        tgt_words = target_text.split()

        if not context_text or not target_text:
            return None
        if len(ctx_words) < self.min_context_words or len(tgt_words) < self.min_target_words:
            return None
        if context_text == target_text:
            return None

        return ContextTargetPair(
            state_id=state_id,
            context_text=context_text,
            target_text=target_text,
            domain=domain,
            context_word_count=len(ctx_words),
            target_word_count=len(tgt_words),
            strategy=strategy,
        )

    def generate_pair(self, state_text: str, state_id: str = "state", domain: str = "unknown") -> Optional[ContextTargetPair]:
        if self.strategy == "span_mask":
            return self._generate_span_mask_pair(state_text, state_id, domain)
        return self._generate_split_pair(state_text, state_id, domain)

    def generate_pairs_batch(
        self,
        records: List[Dict[str, Any]],
    ) -> Tuple[List[ContextTargetPair], ViewGenerationStats]:
        """Processes a list of state records into context-target pairs with diagnostics."""
        pairs: List[ContextTargetPair] = []
        reasons: Dict[str, int] = {
            "too_short": 0,
            "insufficient_context": 0,
            "insufficient_target": 0,
            "identical_views": 0,
            "empty_view": 0,
        }

        identical_count = 0
        overlap_count = 0
        empty_ctx_count = 0
        empty_tgt_count = 0

        for r in records:
            text = r.get("state_text", "")
            words = text.strip().split()
            if len(words) < self.min_words_total:
                reasons["too_short"] += 1
                continue

            pair = self.generate_pair(
                state_text=text,
                state_id=r.get("state_id", "unknown"),
                domain=r.get("domain", "unknown"),
            )

            if pair is None:
                reasons["insufficient_target"] += 1
                continue

            if pair.context_text == pair.target_text:
                reasons["identical_views"] += 1
                identical_count += 1
                continue

            if not pair.context_text:
                empty_ctx_count += 1
                reasons["empty_view"] += 1
                continue

            if not pair.target_text:
                empty_tgt_count += 1
                reasons["empty_view"] += 1
                continue

            pairs.append(pair)

        ctx_lens = [p.context_word_count for p in pairs]
        tgt_lens = [p.target_word_count for p in pairs]

        stats = ViewGenerationStats(
            total_states_evaluated=len(records),
            valid_pairs_generated=len(pairs),
            rejected_states_count=len(records) - len(pairs),
            rejection_reasons=reasons,
            avg_context_word_count=float(sum(ctx_lens) / max(len(pairs), 1)),
            avg_target_word_count=float(sum(tgt_lens) / max(len(pairs), 1)),
            identical_pairs_count=identical_count,
            overlapping_pairs_count=overlap_count,
            empty_context_count=empty_ctx_count,
            empty_target_count=empty_tgt_count,
        )

        return pairs, stats
