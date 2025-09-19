"""Tests for the HypothesisEngine."""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from arc_solver.hypothesis import HypothesisEngine
from arc_solver.grid import to_array
from arc_solver.rft import RelationalFrameAnalyzer
from arc_solver.human_reasoning import HumanGradeReasoner


def test_rotation_hypothesis_generation():
    engine = HypothesisEngine()
    inp = to_array([[1, 0], [2, 0]])
    out = np.rot90(inp)
    hyps = engine.generate_hypotheses([(inp, out)])
    assert any(
        h.transformation_type == "rotation" and h.program_sketch[0][1]["k"] == 1
        for h in hyps
    )


def test_color_mapping_hypothesis_generation():
    engine = HypothesisEngine()
    inp = to_array([[1, 2], [1, 2]])
    out = to_array([[2, 3], [2, 3]])
    hyps = engine.generate_hypotheses([(inp, out)])
    assert any(h.transformation_type == "color_swap" for h in hyps)
    h = hyps[0]
    score = engine.test_hypothesis(h, [(inp, out)])
    assert score >= 0


def test_block_row_flip_hypothesis():
    engine = HypothesisEngine()
    inp = to_array([[1, 2], [3, 4]])
    h, w = inp.shape
    factor = 3
    out = np.zeros((h * factor, w * factor), dtype=inp.dtype)
    for br in range(factor):
        block = inp if br % 2 == 0 else np.fliplr(inp)
        for bc in range(factor):
            out[br * h : (br + 1) * h, bc * w : (bc + 1) * w] = block

    hyps = engine.generate_hypotheses([(inp, out)])
    block_hyps = [h for h in hyps if h.transformation_type == "block_row_flip"]
    assert block_hyps, "Expected block_row_flip hypothesis to be generated"

    hypothesis = block_hyps[0]
    score = engine.test_hypothesis(hypothesis, [(inp, out)])
    assert score == 1.0

    test_input = to_array([[4, 5], [6, 7]])
    predicted = engine.apply(hypothesis, test_input)
    assert predicted.shape == out.shape
    assert np.array_equal(predicted[0:2, 0:2], test_input)
    assert np.array_equal(predicted[2:4, 0:2], np.fliplr(test_input))


def test_pattern_stamp_hypothesis():
    engine = HypothesisEngine()
    inp = to_array([[0, 5, 0], [5, 5, 0], [0, 0, 0]])
    h, w = inp.shape
    out = np.zeros((h * h, w * w), dtype=inp.dtype)
    for i in range(h):
        for j in range(w):
            if inp[i, j] != 0:
                out[i * h : (i + 1) * h, j * w : (j + 1) * w] = inp

    hyps = engine.generate_hypotheses([(inp, out)])
    stamp_hyps = [h for h in hyps if h.transformation_type == "pattern_stamp"]
    assert stamp_hyps, "Expected pattern_stamp hypothesis to be generated"

    hypothesis = stamp_hyps[0]
    score = engine.test_hypothesis(hypothesis, [(inp, out)])
    assert score == 1.0

    test_input = to_array([[0, 2, 2], [0, 0, 0], [2, 0, 0]])
    predicted = engine.apply(hypothesis, test_input)
    expected = np.zeros_like(out)
    for i in range(h):
        for j in range(w):
            if test_input[i, j] != 0:
                expected[i * h : (i + 1) * h, j * w : (j + 1) * w] = test_input
    assert np.array_equal(predicted, expected)


def test_sort_rows_hypothesis():
    engine = HypothesisEngine()
    inp = to_array([[2, 1, 3], [5, 4, 4]])
    out = np.sort(inp, axis=1)
    hyps = engine.generate_hypotheses([(inp, out)])
    assert any(h.transformation_type == "sort_rows" for h in hyps)
    hyp = next(h for h in hyps if h.transformation_type == "sort_rows")
    prediction = engine.apply(hyp, inp)
    assert np.array_equal(prediction, out)


def test_align_top_left_hypothesis():
    engine = HypothesisEngine()
    inp = to_array([[0, 0, 0], [0, 0, 4], [0, 4, 4]])
    out = to_array([[4, 0, 0], [4, 4, 0], [0, 0, 0]])
    hyps = engine.generate_hypotheses([(inp, out)])
    assert any(h.transformation_type == "align_top_left" for h in hyps)
    hyp = next(h for h in hyps if h.transformation_type == "align_top_left")
    prediction = engine.apply(hyp, inp)
    assert np.array_equal(prediction, out)


def test_fill_holes_hypothesis():
    engine = HypothesisEngine()
    inp = to_array(
        [
            [0, 0, 0, 0, 0],
            [0, 3, 3, 3, 0],
            [0, 3, 0, 3, 0],
            [0, 3, 3, 3, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    out = to_array(
        [
            [0, 0, 0, 0, 0],
            [0, 3, 3, 3, 0],
            [0, 3, 4, 3, 0],
            [0, 3, 3, 3, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    hyps = engine.generate_hypotheses([(inp, out)])
    fill_hyp = next(h for h in hyps if h.transformation_type == "fill_holes")
    prediction = engine.apply(fill_hyp, inp)
    assert np.array_equal(prediction, out)


def test_fill_regions_by_area_hypothesis():
    engine = HypothesisEngine()
    inp = to_array(
        [
            [2, 2, 2, 2],
            [2, 0, 0, 2],
            [2, 0, 0, 2],
            [2, 2, 2, 2],
        ]
    )
    out = to_array(
        [
            [2, 2, 2, 2],
            [2, 7, 7, 2],
            [2, 7, 7, 2],
            [2, 2, 2, 2],
        ]
    )
    hyps = engine.generate_hypotheses([(inp, out)])
    area_hyp = next(h for h in hyps if h.transformation_type == "fill_regions_by_area")
    prediction = engine.apply(area_hyp, inp)
    assert np.array_equal(prediction, out)


def test_relational_facts_generate_composite_and_inverse():
    analyzer = RelationalFrameAnalyzer()
    inp = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 2, 0],
        ],
        dtype=np.int16,
    )
    out = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 2, 0],
        ],
        dtype=np.int16,
    )

    facts = analyzer.analyze([(inp, out)])

    assert "inverse" in facts and facts["inverse"], "Expected inverse relational facts"
    assert any(f.relation == "opposite_spatial" for f in facts["inverse"])
    assert "composite" in facts and facts["composite"], "Expected composite relational facts"


def test_human_reasoner_generates_relation_translation_hypothesis():
    reasoner = HumanGradeReasoner()
    inp = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 2, 0],
        ],
        dtype=np.int16,
    )
    out = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 2, 0],
        ],
        dtype=np.int16,
    )

    hyps = reasoner.analyze_task([(inp, out)])
    relation_hypotheses = [h for h in hyps if h.metadata and h.metadata.get("type") == "relation_translation"]
    assert relation_hypotheses, "Expected relation_translation hypothesis"

    relation_hypothesis = relation_hypotheses[0]
    predicted = relation_hypothesis.construction_rule(inp)
    assert predicted.shape == out.shape
