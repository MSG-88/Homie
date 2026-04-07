## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-04-07*

### Ready to Build (scored >= 8, validated)

- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-04-07) — Tutorial on semantic search with sentence-transformers; Homie RAG pipeline already needs embeddings â€” this is the go-to lightweight model [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **sentence-transformers/all-mpnet-base-v2** (score: 9, source: huggingface, added: 2026-04-07) — Higher-quality embedding model for Homie RAG; tutorial comparing MiniLM vs mpnet tradeoffs [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **BAAI/bge-m3** (score: 9, source: huggingface, added: 2026-04-07) — State-of-art multilingual embeddings tutorial; top candidate for Homie's RAG embedding backbone [link](https://huggingface.co/BAAI/bge-m3)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-04-07) — Tutorial on reranking for RAG; Homie could add cross-encoder reranking to improve retrieval quality [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **colbert-ir/colbertv2.0** (score: 8, source: huggingface, added: 2026-04-07) — Tutorial on ColBERT late-interaction retrieval; Homie could use for advanced local document search [link](https://huggingface.co/colbert-ir/colbertv2.0)
- **BAAI/bge-small-en-v1.5** (score: 8, source: huggingface, added: 2026-04-07) — Compact English embeddings tutorial; ideal for Homie's resource-constrained local RAG on smaller devices [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **Qwen/Qwen3-0.6B** (score: 8, source: huggingface, added: 2026-04-07) — Tutorial on running small LLMs locally; Homie could integrate Qwen3-0.6B as a lightweight local chat model [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **hexgrad/Kokoro-TTS** (score: 8, source: huggingface, added: 2026-04-07) — Tutorial on local TTS; Homie voice interaction could use Kokoro for high-quality local speech synthesis [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)

### Prototyping (scored >= 7)

- **openai/clip-vit-large-patch14** (score: 7, source: huggingface, added: 2026-04-07) — Tutorial on zero-shot image classification; Homie plugin for local image search/tagging [link](https://huggingface.co/openai/clip-vit-large-patch14)
- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 7, source: huggingface, added: 2026-04-07) — Multilingual embedding tutorial; Homie could use this for non-English RAG support [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

### Evaluating (scored >= 6)

- **Falconsai/nsfw_image_detection** (score: 6, source: huggingface, added: 2026-04-07) — Homie content-safety plugin to filter NSFW images locally before processing [link](https://huggingface.co/Falconsai/nsfw_image_detection)
- **laion/clap-htsat-fused** (score: 6, source: huggingface, added: 2026-04-07) — Homie voice plugin could use audio classification for sound event detection locally [link](https://huggingface.co/laion/clap-htsat-fused)
- **openai/clip-vit-base-patch32** (score: 6, source: huggingface, added: 2026-04-07) — Lighter CLIP variant for tutorials on multimodal AI; Homie local image search plugin [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **mteb/leaderboard** (score: 6, source: huggingface, added: 2026-04-07) — Tutorial on embedding benchmarks; helps Homie users choose the best RAG model [link](https://huggingface.co/spaces/mteb/leaderboard)
- **krisdcosta/edge-llm-bench** (score: 6, source: huggingface, added: 2026-04-07) — Edge LLM benchmarks could inform a tutorial on local model selection and help Homie choose optimal models for device constraints [link](https://huggingface.co/datasets/krisdcosta/edge-llm-bench)

### Discovered (new)

*No items yet.*
