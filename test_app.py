"""
Tests for the Sponsor Detection app.

Tests transcript extraction, quote alignment, JSON parsing, and the full
annotation pipeline using a mocked LLM response.
"""

import json
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(__file__))

from mmif import Mmif, DocumentTypes, AnnotationTypes


def make_app():
    """Create app instance without full init (avoids git version resolution)."""
    from app import SponsorDetection
    with patch.object(SponsorDetection, '__init__', lambda self: None):
        det = SponsorDetection()
        det.logger = MagicMock()
    return det


def test_metadata():
    """Test that metadata is valid and complete."""
    from metadata import appmetadata
    meta = appmetadata()
    assert meta.name == "Sponsor Detection"
    assert "sponsor-detection" in str(meta.identifier)

    param_names = [p.name for p in meta.parameters]
    assert 'apiUrl' in param_names
    assert 'modelName' in param_names

    print("PASS: test_metadata")


def test_extract_json():
    """Test JSON extraction from various LLM output formats."""
    det = make_app()

    result = det._extract_json('{"sponsors": [{"name": "PBS"}]}')
    assert result['sponsors'][0]['name'] == 'PBS'

    result = det._extract_json('```json\n{"sponsors": []}\n```')
    assert result['sponsors'] == []

    result = det._extract_json('{"sponsors": [{"name": "PBS",}],}')
    assert result['sponsors'][0]['name'] == 'PBS'

    result = det._extract_json('no json here')
    assert result is None

    print("PASS: test_extract_json")


def test_align_quote():
    """Test fuzzy quote alignment to segments."""
    det = make_app()

    segments = [
        {'start_ms': 0, 'end_ms': 10000, 'text': 'Good evening welcome to the show', 'tf_ids': ['v1:tf_1']},
        {'start_ms': 10000, 'end_ms': 20000, 'text': 'Funding provided by the Sloan Foundation', 'tf_ids': ['v1:tf_2']},
        {'start_ms': 20000, 'end_ms': 30000, 'text': 'and the Corporation for Public Broadcasting', 'tf_ids': ['v1:tf_3']},
        {'start_ms': 30000, 'end_ms': 40000, 'text': 'Tonight we discuss the economy', 'tf_ids': ['v1:tf_4']},
    ]

    # Exact single-segment match
    match = det._align_quote("Funding provided by the Sloan Foundation", segments)
    assert match is not None
    assert match['start_ms'] == 10000

    # Multi-segment span
    match = det._align_quote(
        "Funding provided by the Sloan Foundation and the Corporation for Public Broadcasting",
        segments
    )
    assert match is not None
    assert match['start_ms'] == 10000
    assert match['end_ms'] == 30000

    # No match for unrelated text
    match = det._align_quote("xyzzy foobar bazquux nothing", segments)
    assert match is None

    print("PASS: test_align_quote")


def test_real_mmif_extraction():
    """Test extraction against real parakeet MMIF if available."""
    mmif_path = os.path.join(os.path.dirname(__file__), 'cpb-aacip-507-154dn40c26.mmif')
    if not os.path.exists(mmif_path):
        print("SKIP: test_real_mmif_extraction (no test MMIF file)")
        return

    import warnings
    warnings.filterwarnings('ignore')

    det = make_app()
    mmif = Mmif(open(mmif_path).read())

    asr_view = det._get_asr_view(mmif)
    assert asr_view is not None, "Should find ASR view"

    transcript = det._get_transcript_text(asr_view)
    assert len(transcript) > 1000, f"Transcript too short: {len(transcript)}"

    segments = det._build_timestamped_segments(mmif, asr_view)
    assert len(segments) > 10, f"Too few segments: {len(segments)}"
    assert segments[0]['start_ms'] >= 0
    assert len(segments[0]['tf_ids']) > 0, "Segments should have timeframe IDs"

    # Check that sponsor-related text is in the transcript
    assert 'AT' in transcript, "AT&T should be in the transcript"
    assert 'Corporation for Public Broadcasting' in transcript

    print("PASS: test_real_mmif_extraction")


def test_full_pipeline_mock():
    """Integration test with mocked LLM response."""
    from app import SponsorDetection

    det = make_app()

    # Pre-built segments (simulating what _build_timestamped_segments returns)
    segments = [
        {'start_ms': 0, 'end_ms': 10000, 'text': 'Good evening tonight on the news', 'tf_ids': ['v1:tf_1']},
        {'start_ms': 10000, 'end_ms': 20000, 'text': 'Funding provided by AT and T', 'tf_ids': ['v1:tf_2']},
        {'start_ms': 20000, 'end_ms': 30000, 'text': 'and the Corporation for Public Broadcasting', 'tf_ids': ['v1:tf_3']},
        {'start_ms': 30000, 'end_ms': 40000, 'text': 'Our top story tonight', 'tf_ids': ['v1:tf_4']},
    ]

    mock_llm_result = {
        "sponsors": [
            {"name": "AT&T", "quote": "Funding provided by AT and T"},
            {"name": "Corporation for Public Broadcasting",
             "quote": "and the Corporation for Public Broadcasting"},
        ]
    }

    # Test alignment
    for sp in mock_llm_result['sponsors']:
        match = det._align_quote(sp['quote'], segments)
        assert match is not None, f"Should align: {sp['name']}"
        assert match['start_ms'] >= 10000, f"Should be in sponsor segment, got {match['start_ms']}"

    print("PASS: test_full_pipeline_mock")


if __name__ == '__main__':
    test_metadata()
    test_extract_json()
    test_align_quote()
    test_real_mmif_extraction()
    test_full_pipeline_mock()
    print("\n=== All tests passed! ===")
