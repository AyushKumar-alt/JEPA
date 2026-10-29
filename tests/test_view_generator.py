"""
Unit tests for context-target view generator.
Tests both 'span_mask' (preferred data2vec/JEPA design) and 'sentence_split'.
"""

import pytest
from src.view_generator import ViewGenerator, ContextTargetPair


def test_view_generation_span_masking():
    vg = ViewGenerator(strategy="span_mask", min_words_total=8, mask_ratio=0.25, seed=42)
    state = "The customer requested a replacement item because the previous package was damaged in transit."
    pair = vg.generate_pair(state, state_id="test_span")

    assert pair is not None
    assert isinstance(pair, ContextTargetPair)
    assert pair.strategy == "span_mask"
    assert pair.target_text == state  # Target is full state
    assert "[MASK]" in pair.context_text  # Context is masked
    assert pair.context_text != pair.target_text
    assert len(pair.context_text) > 0
    assert len(pair.target_text) > 0


def test_view_generation_valid_pair_sentence_split():
    vg = ViewGenerator(strategy="sentence_split", min_words_total=8, context_ratio=0.6)
    state = "The customer requested a replacement item. The previous package was damaged in transit. Delivery was delayed."
    pair = vg.generate_pair(state, state_id="test_1")

    assert pair is not None
    assert isinstance(pair, ContextTargetPair)
    assert pair.context_text != pair.target_text
    assert len(pair.context_text) > 0
    assert len(pair.target_text) > 0
    assert pair.context_word_count >= 3
    assert pair.target_word_count >= 3


def test_view_generation_short_state_handling():
    vg = ViewGenerator(min_words_total=8)
    short_state = "Hello world."
    pair = vg.generate_pair(short_state, state_id="test_short")
    assert pair is None


def test_view_generation_deterministic():
    vg1 = ViewGenerator(strategy="span_mask", min_words_total=8, seed=42)
    vg2 = ViewGenerator(strategy="span_mask", min_words_total=8, seed=42)
    state = "First action performed in the workflow. Second action followed immediately afterwards."
    pair1 = vg1.generate_pair(state, state_id="fixed")
    pair2 = vg2.generate_pair(state, state_id="fixed")
    assert pair1.context_text == pair2.context_text
    assert pair1.target_text == pair2.target_text


def test_view_generation_batch_stats():
    vg = ViewGenerator(strategy="span_mask", min_words_total=8)
    records = [
        {"state_id": "1", "state_text": "A valid sentence with enough words to split into context and target easily.", "domain": "test"},
        {"state_id": "2", "state_text": "Too short.", "domain": "test"},
        {"state_id": "3", "state_text": "Another long state text describing system logs and network throughput details.", "domain": "test"},
    ]
    pairs, stats = vg.generate_pairs_batch(records)
    assert len(pairs) == 2
    assert stats.rejected_states_count == 1
    assert stats.identical_pairs_count == 0
    assert stats.empty_context_count == 0
    assert stats.empty_target_count == 0
