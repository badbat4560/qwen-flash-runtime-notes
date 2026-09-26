# Qwen Flash-Next: runtime compatibility notes

Companion material for a deployment write-up covering checkpoint loading, B12x integration, and memory tuning. All publication text is in English.

This is a collection of extracted compatibility changes and evidence notes. It does not reconstruct the full custom serving image or provide a turnkey deployment. Additional attention, offload, MTP, and scheduler integration existed in the source environment and is not included here.

## Contents

- [Article](docs/article.md): the full narrative and measurement limits.
- [B12x wrapper](patches/b12x_compat.py): invalid-route handling and compatible workspace reuse.
- [PLE loader patch](patches/ple-unsplit.patch): support for an unsplit embedding tensor and short checkpoint keys.
- [Tests](tests/test_b12x_compat.py): CPU tests of the wrapper using a fake FlashInfer class.
- [Baseline parameters](examples/baseline.json): the initial bfloat16 KV-cache configuration.
- [Integration notes](docs/integration.md): assumptions, patch application, and validation boundaries.
- [Results](docs/results.md): what the historical concurrent test did and did not measure.

## Run the wrapper tests

Use a separate environment with a Python-compatible PyTorch installation:

```bash
python -m venv .venv
# Activate the environment using the command appropriate for your shell.
python -m pip install -r requirements-test.txt
python -m pytest -q
```

These tests need PyTorch but do not need CUDA, model weights, or a FlashInfer installation. They replace the imported FlashInfer wrapper with a test double. Passing them does not validate GPU kernels or end-to-end generation.

## Credits

Big respect to **windowsxp811203** for the quality of the work and detailed reports.

The runtime discussed here builds on vLLM, FlashInfer, and PyTorch. Upstream code and model artifacts retain their respective licenses. The PLE change is distributed as a small diff rather than a full copy of the upstream module. No blanket license or ownership claim over upstream components is made by this collection.
