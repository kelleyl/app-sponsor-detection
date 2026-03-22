# TODO

## Deployment
- [ ] Test containerized deployment (Containerfile not yet tested)
- [ ] Decide how to run the LLM server alongside the app container (sidecar, shared pod, external service)
- [ ] Test with `transformers serve` and vllm in addition to ollama
- [ ] Test with Qwen3.5-4B once ollama or transformers serve compatibility is resolved

## MMIF Output
- [ ] Improve output representation — current output is a JSON TextDocument with sponsor name + time range, linked to source TimeFrames via Alignment. Consider:
  - Should sponsors be represented as a different annotation type?
  - Should there be a single summary TextDocument listing all sponsors, or one per sponsor (current)?
  - How to represent the sponsor mention span within the transcript (character offsets, token IDs)?
  - Should the output include the full quoted text from the transcript?

## Evaluation
- [ ] Run on more NewsHour episodes to assess recall/precision across eras
- [ ] Test on non-NewsHour PBS programs (Nova, American Experience, Frontline)
- [ ] Test on programs without sponsors to check false positive rate
