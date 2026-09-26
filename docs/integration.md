# Integration boundaries

## B12x wrapper

The extracted wrapper patches the private import path `flashinfer.fused_moe.cute_dsl.b12x_moe.B12xMoEWrapper`. Check your installed version before use: private APIs can change, and newer upstream versions may already provide shared-workspace support.

Copy `patches/b12x_compat.py` to an importable module location in the development runtime. The integration hook is:

```python
import b12x_compat
b12x_compat.install()
```

Call it in each relevant worker process before wrapper instances are created and before CUDA graph capture. Merely copying the file into an image does not activate it. This bundle intentionally supplies no automatic startup hook: that hook must fit the worker lifecycle of the selected runtime.

The wrapper expects route tensors as the keyword arguments `token_selected_experts` and `token_final_scales`. It maps invalid expert IDs to zero and zeroes their routing weights. It preserves valid routes.

Workspace reuse includes static, dynamic, and output buffers. It assumes compatible wrappers execute sequentially without overlapping use of those buffers. Its compatibility key reflects the source runtime; review any additional backend attributes introduced by a different version. Shared buffers persist in a process-local cache.

Reference: [FlashInfer fused MoE API](https://docs.flashinfer.ai/api/fused_moe.html). Current upstream supports explicit shared buffers; assess that API before adopting a historical monkey patch.

## PLE loader diff

The diff targets a module named `ple_layer.py`. It accepts full-table tensors with an empty shard suffix as well as the short key prefix, validates tensor dimensions, and calls an existing copy helper with the relevant partition boundaries.

In a disposable source checkout, from the directory containing the target module:

```bash
git apply --check /path/to/ple-unsplit.patch
git apply /path/to/ple-unsplit.patch
```

The paths above are placeholders. This diff depends on the surrounding module, including its existing copy helper; it is not an independent loader implementation. Compare against the intended base first. Do not force a failed patch application.

The original source image used for this narrow PLE diff was recorded as `vllm/vllm-openai:qwen38-flash-next` with digest `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`. This identifies the recorded initial base, not the later full B12x stack. Registry availability has not been rechecked.

## Attention and other runtime changes

The initial run overrode twelve attention-layer entries from `qwen_sparse_attention` to `full_attention`. This bundle describes that historical step but supplies no automatic transformation: successful loading alone did not validate computational equivalence or answer quality.

The complete later runtime also included graph-shape, prefill, MTP, and scheduler choices. Exact compatible versions and all associated patches are not reconstructed here. The baseline JSON is documentation, not a vLLM configuration file to pass directly to the CLI.

## Validation before runtime use

Validate backend selection, loading, real generation, overlapping requests, and long-context behavior in an isolated runtime. CPU tests of the wrapper cannot establish numerical quality, safe CUDA graph behavior, or suitability for a different execution schedule. No production container was changed while preparing this bundle.
