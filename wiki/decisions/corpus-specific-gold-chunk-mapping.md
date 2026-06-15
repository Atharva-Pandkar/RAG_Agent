## Corpus-Specific Gold Chunk Mapping
- Date: 2026-06-15
- Context: Gold evidence is defined as text quotes, but retrieval metrics need chunk IDs. Chunk boundaries change when chunking strategy or size changes.
- Decision: Never mutate `golden_set.json` with chunk IDs. Instead, run `populate_gold_chunks.py --corpus <path>` to produce a sidecar file (e.g. `golden_set_with_chunks_fixed_size_512_50.json`) referenced via `golden_set_override` in experiment configs.
- Alternatives Considered: Store chunk IDs directly in canonical golden set; manual chunk ID annotation per question.
- Impact: Re-chunking requires re-running population and updating config overrides. Evidence matching uses a 60-char anchor substring to tolerate minor formatting differences.
