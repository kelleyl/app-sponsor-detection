"""
Sponsor Detection App

Detects sponsor mentions in video programs by analyzing ASR transcript output.

1. Extracts transcript text and timestamped segments from upstream ASR (Whisper/Parakeet)
2. Sends the full transcript to an LLM via an OpenAI-compatible API
   (works with transformers serve, vllm, sglang, ollama, etc.)
3. LLM identifies sponsors and returns quoted text
4. Aligns quotes back to timestamped segments to produce TimeFrame-linked annotations

Outputs:
- TextDocument: JSON with sponsor name and aligned time range
- Alignment: Links each sponsor detection to the matching transcript segment
"""

import argparse
import logging
import json
import re
from difflib import SequenceMatcher
from typing import List, Dict, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

from clams import ClamsApp, Restifier
from mmif import Mmif, View, AnnotationTypes, DocumentTypes


class SponsorDetection(ClamsApp):

    def __init__(self):
        super().__init__()

    def _appmetadata(self):
        pass

    def _get_asr_view(self, mmif: Mmif) -> Optional[View]:
        """Find the ASR view (contains both TextDocument and Token)."""
        for v in mmif.get_all_views_contain(AnnotationTypes.Token):
            tds = list(v.get_annotations(DocumentTypes.TextDocument))
            if tds:
                return v
        return None

    def _get_transcript_text(self, view: View) -> str:
        """Get the full transcript text from an ASR view's TextDocument."""
        for td in view.get_annotations(DocumentTypes.TextDocument):
            text_val = td.get_property('text')
            t = text_val.value if hasattr(text_val, 'value') else str(text_val)
            if len(t) > 50:
                return t
        return ""

    def _build_timestamped_segments(self, mmif: Mmif, view: View) -> List[Dict]:
        """
        Build ~10s timestamped text segments from Token+TimeFrame+Alignment.
        Each segment has: start_ms, end_ms, text, tf_ids (for alignment output).
        """
        tok_time = {}
        for al in view.get_annotations(AnnotationTypes.Alignment):
            src_id = al.get_property('source')
            tgt_id = al.get_property('target')
            try:
                src_obj = mmif[src_id]
                tgt_obj = mmif[tgt_id]
                if (src_obj.at_type == AnnotationTypes.TimeFrame
                        and tgt_obj.at_type == AnnotationTypes.Token):
                    s = src_obj.get_property('start')
                    e = src_obj.get_property('end')
                    if s is not None and e is not None:
                        tok_time[tgt_id] = (int(s), int(e), src_id)
            except Exception:
                pass

        sorted_toks = []
        for tok in view.get_annotations(AnnotationTypes.Token):
            if tok.id in tok_time:
                ms_s, ms_e, tf_id = tok_time[tok.id]
                word = tok.get_property('word', '') or tok.get_property('text', '')
                sorted_toks.append((ms_s, ms_e, word, tf_id))
        sorted_toks.sort(key=lambda x: x[0])

        if not sorted_toks:
            return []

        WINDOW_MS = 10000
        segments = []
        cur_start = sorted_toks[0][0]
        cur_words = []
        cur_tf_ids = []
        for ms_s, ms_e, word, tf_id in sorted_toks:
            if ms_s - cur_start > WINDOW_MS and cur_words:
                segments.append({
                    'start_ms': cur_start,
                    'end_ms': ms_s,
                    'text': ' '.join(cur_words),
                    'tf_ids': list(set(cur_tf_ids)),
                })
                cur_words = []
                cur_tf_ids = []
                cur_start = ms_s
            cur_words.append(word)
            cur_tf_ids.append(tf_id)
        if cur_words:
            segments.append({
                'start_ms': cur_start,
                'end_ms': sorted_toks[-1][1],
                'text': ' '.join(cur_words),
                'tf_ids': list(set(cur_tf_ids)),
            })

        return segments

    def _query_llm(self, transcript_text: str, api_url: str, model_name: str) -> Dict:
        """
        Query an OpenAI-compatible API to detect sponsors in the transcript.
        Works with transformers serve, vllm, sglang, ollama, etc.
        """
        prompt = (
            'Identify any sponsors, funders, or underwriters mentioned in this '
            'TV program transcript. Look for phrases like "made possible by", '
            '"brought to you by", "funded by", or mentions of corporate/foundation '
            'names in a sponsorship context.\n\n'
            f'TRANSCRIPT:\n{transcript_text}\n\n'
            'For each sponsor found, quote the EXACT text from the transcript '
            'that mentions this sponsor. Return JSON:\n'
            '{"sponsors": [{"name": "Sponsor Name", "quote": "exact quote from transcript"}]}\n'
            'If none found: {"sponsors": []}'
        )

        payload = json.dumps({
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.8,
        }).encode()

        url = f"{api_url.rstrip('/')}/v1/chat/completions"
        req = Request(url, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
        })

        try:
            with urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())
        except URLError as e:
            self.logger.error(f"LLM API request failed: {e}")
            return {"sponsors": []}

        raw_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = result.get("usage", {})
        self.logger.info(
            f"LLM response: {usage.get('completion_tokens', '?')} tokens, "
            f"prompt={usage.get('prompt_tokens', '?')}"
        )
        self.logger.debug(f"LLM raw output: {raw_text}")

        return self._extract_json(raw_text) or {"sponsors": []}

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response."""
        try:
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                return json.loads(json_str)
        except Exception as e:
            self.logger.debug(f"JSON extraction error: {e}")
        return None

    def _align_quote(self, quote: str, segments: List[Dict]) -> Optional[Dict]:
        """Fuzzy-match a sponsor quote back to timestamped segments."""
        quote_lower = quote.lower().strip()
        best_score = 0
        best_seg = None

        for i, seg in enumerate(segments):
            ratio = SequenceMatcher(None, quote_lower, seg['text'].lower()).ratio()
            if ratio > best_score:
                best_score = ratio
                best_seg = seg

            for j in range(i + 1, min(i + 4, len(segments))):
                combined = ' '.join(segments[k]['text'] for k in range(i, j + 1)).lower()
                ratio = SequenceMatcher(None, quote_lower, combined).ratio()
                if ratio > best_score:
                    best_score = ratio
                    best_seg = {
                        'start_ms': segments[i]['start_ms'],
                        'end_ms': segments[j]['end_ms'],
                        'text': ' '.join(segments[k]['text'] for k in range(i, j + 1)),
                        'tf_ids': sum((segments[k]['tf_ids'] for k in range(i, j + 1)), []),
                    }

        if best_score > 0.5 and best_seg:
            return best_seg
        return None

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        """Main annotation method."""
        api_url = parameters.get('apiUrl', 'http://localhost:8000')
        model_name = parameters.get('modelName', 'Qwen/Qwen3.5-4B')

        # Find ASR view
        asr_view = self._get_asr_view(mmif)
        if not asr_view:
            self.logger.error("No ASR view with Token+TextDocument found in MMIF")
            return mmif

        transcript = self._get_transcript_text(asr_view)
        if not transcript:
            self.logger.error("No transcript text found in ASR view")
            return mmif

        segments = self._build_timestamped_segments(mmif, asr_view)
        self.logger.info(f"Transcript: {len(transcript)} chars, {len(segments)} segments")

        # Query LLM
        result = self._query_llm(transcript, api_url, model_name)
        sponsors = result.get("sponsors", [])
        self.logger.info(f"LLM found {len(sponsors)} sponsor mention(s)")

        # Create output view
        new_view = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(DocumentTypes.TextDocument)
        new_view.new_contain(AnnotationTypes.Alignment)

        # Deduplicate by name
        seen = {}
        for sp in sponsors:
            name = sp.get('name', '').strip()
            if not name:
                continue
            name_key = name.lower()
            quote = sp.get('quote', '')
            matched_seg = self._align_quote(quote, segments) if quote else None

            if name_key not in seen or (matched_seg and not seen[name_key].get('seg')):
                seen[name_key] = {'name': name, 'quote': quote, 'seg': matched_seg}

        # Create annotations
        for name_key, det in seen.items():
            sponsor_json = {'sponsor': det['name']}
            seg = det['seg']
            if seg:
                sponsor_json['startMs'] = seg['start_ms']
                sponsor_json['endMs'] = seg['end_ms']

            td = new_view.new_textdocument(
                text=json.dumps(sponsor_json),
                mime='application/json'
            )

            if seg:
                for tf_id in seg.get('tf_ids', [])[:5]:
                    al = new_view.new_annotation(AnnotationTypes.Alignment)
                    al.add_property("source", tf_id)
                    al.add_property("target", td.long_id)

        self.logger.info(f"Output: {len(seen)} unique sponsor(s)")
        return mmif


def get_app():
    """Factory function for app instantiation."""
    return SponsorDetection()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")

    parsed_args = parser.parse_args()

    app = get_app()
    http_app = Restifier(app, port=int(parsed_args.port))
    if parsed_args.production:
        http_app.serve_production()
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
