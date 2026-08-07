```markdown
halyk-ai-challenge/
├── data/
│   ├── raw/                 # PDFs, master_ledger_2025.csv
│   ├── interleaved/         # Markdown-converted docs, mapped ledger chunks
│   └── outputs/             # submission.json
├── models/
│   ├── base/                # Llama-3.1-8B-Instruct-GGUF or Mistral-Nemo-12B
│   ├── adapters/            # LoRA/QLoRA weights if fine-tuned
│   └── tokenizer/
├── src/
│   ├── doc_processor/
│   │   ├── layout_parser.py # Uses Docling/Marker for "Dirty" PDFs to Markdown
│   │   └── id_mapper.py     # Links PDFs to scenario_id via Account/Company names
│   ├── llm_engine/
│   │   ├── inference.py     # Local inference logic (llama.cpp or vLLM)
│   │   ├── prompts.py       # Structured output templates (JSON extraction)
│   │   └── finetune.py      # Unsloth-based SFT/QLoRA script
│   ├── finance_engine/
│   │   ├── duck_db_client.py# High-speed SQL queries on the CSV
│   │   ├── calculator.py    # Python math for ratios/aggregates
│   │   └── classifier.py    # Regex/Dictionary-based transaction labeling
│   └── orchestrator/
│       ├── agent.py         # Main loop: Extracted Rule -> Python Math -> Verdict
│       └── evidence_logic.py# Logic to identify specific breaching txn_ids
├── notebooks/
│   └── 01_baseline_extraction.ipynb # Testing local LLM extraction accuracy
├── requirements.txt
└── main.py
```
