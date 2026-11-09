# Paper2Perf

## Virtual Environment

Use Python >=3.12. Currently, only Python 3.14 has been tested.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Packages

Install the CPU PyTorch build first, then the project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

Configure the model provider. The framework has been tested on `deepseek/deepseek-v4.1-flash-1017` through OpenRouter.

```bash
cp .env.example .env
# Edit .env and set PAPER2PERF_MODEL and PAPER2PERF_API_KEY.

paper2perf doctor
```

## Start the Chat

Run from the directory where the A100 whitepaper path resolves:

```bash
source .venv/bin/activate

paper2perf --data-dir demo-data chat \
  --documents ./nvidia-ampere-architecture-whitepaper.pdf
```

Exit the chat with:

```text
/quit
```

## Demo Prompts

### 1. Document question

```text
According to this document, what dense FP16 Tensor Core throughput and HBM2 memory bandwidth are documented for the A100 SXM4? Cite the pages.
```

### 2. Simulation

```text
Simulate C = A × B on the A100 SXM4 described in this document. A and B are both 256 × 256. Use dense FP16 with no bias. Preserve the documented memory hierarchy and disclose all modeling assumptions.
```
