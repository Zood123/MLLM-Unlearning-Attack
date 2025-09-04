# MLLM-Unlearning-Attack

This repository provides code for conducting **adversarial attacks on MLLMs**.  

---


## Setup

### Prerequisites
- Python 3.8+


### Environment Variables
Before running, configure Hugging Face authentication:

```bash
export HF_HOME=path/to/huggingface/cache
export HF_TOKEN=your_hf_token_here
```

---

## Repository Structure

- `attack_utils/` — Utility functions and scripts for adversarial attack generation.  
- `llava_llama_2/` — Code specific to LLaVA-LLaMA-2 model setup.  
- `model/` — Scripts and configurations for loading and handling models.  
- `llava_vlm_attack.py` — Main entry point for launching adversarial attacks.  
- `utils.py` — General utility functions.  
- `test.sh` — Example shell script for running attacks.  
- `README.md` — Project documentation.

