# Quick Start: LLM Integration

## 1. Install Dependencies (One-time)

```bash
pip install transformers>=4.36.0 torch>=2.1.0 accelerate>=0.25.0 bitsandbytes>=0.41.0
```

## 2. Enable LLM in Config

Edit `solver_config.json`:

```json
{
  "use_llm_reasoning": true,
  "llm_model_name": "microsoft/Phi-3-mini-4k-instruct"
}
```

## 3. Use in Code

```python
from arc_solver.enhanced_search import EnhancedSearch

# With LLM enabled
search = EnhancedSearch(
    llm_config={
        'use_llm_reasoning': True,
        'llm_model_name': 'microsoft/Phi-3-mini-4k-instruct'
    }
)

# Solve task (LLM will guide search strategy)
programs = search.synthesize_enhanced(train_pairs)

# Check LLM recommendations
stats = search.get_search_statistics()
print(stats['meta_reasoning'])
```

## 4. Run Tests

```bash
python test_llm_integration.py
```

## Alternative Models

```python
# Qwen2.5-3B (smaller, faster)
llm_config = {'llm_model_name': 'Qwen/Qwen2.5-3B-Instruct'}

# Llama-3.2-3B (latest)
llm_config = {'llm_model_name': 'meta-llama/Llama-3.2-3B-Instruct'}
```

## Disable LLM

```python
search = EnhancedSearch(llm_config={'use_llm_reasoning': False})
```

## What the LLM Does

- Analyzes task features, similar episodes, RFT facts
- Recommends best search strategy
- Suggests operation sequences
- Provides confidence scores

## Performance

- Memory: ~2GB with 4-bit quantization
- Speed: ~1-2 seconds per task
- First run: Downloads model (~2GB, one-time)

See `LLM_INTEGRATION.md` for full documentation.
