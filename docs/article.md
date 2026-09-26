# Running Qwen Flash-Next NVFP4 in vLLM: PLE Loading, B12x Fixes, and Stable Inference

Companion code: [PLE and B12x compatibility patches, tests, and deployment notes](https://github.com/hf4560/qwen-flash-runtime-notes). The repository contains extracted fixes and validation notes, not a complete runtime image.

The model loaded. The health endpoint returned a successful response. A few requests completed. Then concurrent traffic arrived, and the inference process ran out of GPU memory.

That was the most instructive part of bringing Qwen Flash-Next NVFP4 online. There was no single switch that made the deployment work. We had to get a large checkpoint onto disk, adapt a loader to its tensor layout, and find a memory configuration that survived actual generation. Each milestone removed one obstacle while exposing the next.

The eventual result was a server configured for 131,072 tokens of context, with a limit of 16 sequences, that completed three rounds of 16 concurrent requests without a recorded container restart. Getting there required giving up some of our initial context and concurrency targets—and being precise about what the successful test actually demonstrated.

Before getting into the details, a big shout-out to **windowsxp811203**. The quality of the work and the detailed reports deserve real credit. It is great to see someone put that much care into both the result and its documentation.

This account follows the initial bfloat16 KV-cache baseline and the additional B12x integration work needed by the developed serving stack. The measured 48-request test belongs to the initial baseline. The later B12x work is described separately, without transferring that test result to a different runtime. The loader and backend patches were required independently of KV-cache quantization.

The first practical problem was simply getting the checkpoint onto the machine. The recorded download comprised 31 files, approximately 186.4 GB in total. A single PLE shard accounted for roughly 102.4 GB. With artifacts of that size, storage preparation and download recovery were part of the deployment work rather than incidental setup.

We provisioned a dedicated volume with an ext4 filesystem and used the Hugging Face CLI with eight download workers. The transfer ran through a local proxy. At one point, progress stalled at around 14 GB. After restarting, the reported progress fell to approximately 4.7 GB before the transfer resumed at about 93 MiB/s.

The practical consequence was extra transfer time and uncertainty about completion. A progress counter was useful for watching the download, but it was not sufficient evidence that the checkpoint was ready. We waited for the final shard to finish before moving on to loading.

It is also important to keep disk size separate from GPU residency. The total checkpoint size was not a prediction of the amount of VRAM the process would consume. This deployment used PLE CPU offload, so some of the storage and loading work belonged on the host side. Later failures still depended on the memory available on the GPU during generation.

Before replacing the existing model server, we saved its container configuration and retained a rollback target. That gave the experiments a defined starting point and preserved a route back if the new model failed. It also prevented the previous deployment from becoming another variable while we were changing the new one.

The initial runtime was a model-specific vLLM container image. It provided a starting point, but it did not accept this checkpoint without local changes. Two incompatibilities surfaced before inference tuning could begin.

The first was in the attention configuration. Twelve layer entries used `qwen_sparse_attention`, while the runtime path we were using expected `full_attention`. We supplied a local configuration override that changed those entries.

That was an operational compatibility change, and it has an important limit: getting past configuration parsing does not establish that the two settings are computationally equivalent. This work did not include a controlled comparison of output quality or attention behavior before and after the override. The evidence supports the narrower claim that the override allowed this runtime to proceed with loading.

The second incompatibility involved the layout of the PLE embedding table. The loader expected checkpoint shards with numeric suffixes. The checkpoint instead exposed the complete table through a tensor name whose shard index was empty. There was also a naming mismatch: the loader path had to recognize both the longer prefixed form and the shorter form actually encountered during loading.

The distinction sounds small until the tensor behind the name is enormous. A loader cannot safely infer that an unrecognized name represents an interchangeable piece of the model. It needs to identify the layout and validate the shape before copying the data.

Our fix went through three iterations. The first recognized the empty suffix and tried to hand the tensor to the ordinary loading path. That was insufficient. The second added explicit handling for the complete table, including shape validation, but recognized only the longer prefix. The third also handled the shorter name used by the checkpoint. That was the version that loaded successfully.

The successful path checked the incoming tensor against the expected full embedding dimensions. It then used the embedding-copy helper with a checkpoint offset of zero and the relevant embedding partition boundaries. Numbered shards retained their existing handling, and other weights continued through the regular loader.

The shape check was an essential part of the change. Supporting an additional serialization layout should not mean accepting arbitrary tensor dimensions just because the name looks familiar. A mismatch should fail at loading, where the error can be explained, rather than being allowed to become an inference problem later.

We packaged the patch as a small layer over the original container image, replacing a single loader module. Keeping that change narrow made it easier to understand what differed between the starting runtime and the working one. The rest of the deployment could then be investigated without treating the entire runtime as a custom fork.

The PLE loader was only one part of the runtime work. The serving stack subsequently required a separate set of fixes at the vLLM–FlashInfer boundary to use the B12x MoE path on our Blackwell SM120 GPU. This is a separate stage from the initial concurrency test reported below; its patches should not be retroactively attributed to that earlier run.

That distinction changes how the deployment should be described. A single loader replacement explains the initial checkpoint-loading fix. The more developed serving image also carries backend integration, speculative-decoding, offload, and scheduling changes. Reproducing that stack requires preserving the complete working runtime, rather than assuming that a stock image plus the PLE patch is equivalent.

The B12x work addressed two very different problems: invalid expert routes and duplicated workspace allocations. We changed the integration around the backend. We did not rewrite the B12x CUDA kernels.

During dummy profiling and padding, the integration could pass an expert ID of `-1`. That value represented an inactive route on the caller's side, but it could reach a path that treated it as an expert index. Our deployment notes associate this failure with `cudaErrorIllegalAddress`, GPU-process failures, and Xid 31 reports.

The wrapper now validates each selected expert ID before calling the original backend. A route is valid only when its ID is nonnegative and below the expert count. An invalid route is redirected to expert zero and its routing weight is set to zero. Valid routes retain both their IDs and weights.

The essential operation is small:

```python
valid = (expert_ids >= 0) & (expert_ids < num_experts)
safe_ids = where(valid, expert_ids, 0)
safe_weights = where(valid, routing_weights, 0)
```

Redirecting an invalid ID alone would be wrong: it would turn padding into a real contribution from expert zero. Clearing the routing weight is what preserves the intended zero contribution of an inactive route. The wrapper also catches IDs above the valid range, although that defensive handling is not a substitute for diagnosing an upstream routing bug if one appears outside padding or profiling.

The saved unit test makes the transformation explicit. Given IDs `[-1, 7, 512]` for a model with 512 experts, the wrapper passes `[0, 7, 0]`. Given routing weights `[0.4, 0.5, 0.6]`, it passes `[0.0, 0.5, 0.0]`. That verifies the wrapper's treatment of invalid routes; it is not, by itself, a numerical validation of the complete GPU inference path.

The second problem was workspace ownership. Our deployment notes describe allocations of roughly 400 MiB per MoE layer across 48 layers. At that scale, allocating an independent workspace for every layer can consume approximately 18.75 GiB before considering the rest of the runtime. This is an estimate from the reported per-layer allocation, not a separately measured memory saving.

Those layers execute sequentially in the execution pattern for which the patch was built. Compatible wrappers can therefore reuse scratch storage instead of reserving a separate copy for every layer.

The implementation caches three buffers: the static workspace, the dynamic workspace, and the MoE output buffer. The cache key includes expert count, top-k, hidden and intermediate dimensions, maximum token count, local expert count, output dtype, device, activation, quantization mode, and source format. Wrappers with matching keys reuse the same buffers; a different key triggers a separate allocation.

Compatibility and execution order both matter. Matching dimensions alone would not make shared writable storage safe for overlapping executions. In this deployment, reuse relies on the sequential layer execution pattern. A change to scheduling or stream behavior would require checking that assumption again.

The saved workspace test verifies that two identically configured wrappers allocate once and share the same buffer objects. That is useful evidence about the Python integration. It does not replace a runtime test under CUDA graphs and real traffic.

This design is also reflected in the current [FlashInfer B12x API documentation](https://docs.flashinfer.ai/api/fused_moe.html), which exposes optional shared static, dynamic, and output buffers for compatible wrappers. That describes the current upstream interface; it does not establish that the runtime version used in our deployment already provided an equivalent integration.

Why pursue B12x at all? Our deployment notes identify Marlin as the fallback and B12x as the preferred native execution path for the NVFP4 MoE workload on this GPU. This article does not include a controlled comparison between the two, so we do not attach a speedup number or a claim of identical output quality to that choice. Selecting a backend does not, by itself, requantize the checkpoint, but numerical equivalence still needs measurement.

The surrounding runtime also needed coordinated choices for CUDA graph shapes, prefill budget, and MTP fallback. The recorded constraints included the `2^31 - 1` memref limit in the affected path, autotuner allocations, and a narrow VRAM margin. These were additional integration constraints, not all manifestations of the same memory bug. The memref limit in particular should not be read as a universal limit on GPU memory or model context.

This is why the custom development runtime mattered. Sanitizing routes addressed invalid indexing. Workspace reuse addressed duplicate buffers. Graph, prefill, and speculative-decoding choices addressed other execution constraints. The PLE loader change solved a separate checkpoint-format problem. Replacing the full image with a stock runtime would discard that combination of fixes in the environment described here.

The resulting stack combines separate responsibilities:

```text
NVFP4 MoE experts
  -> FlashInfer B12x
  -> route validation and compatible workspace reuse

PLE n-gram table
  -> CPU offload and checkpoint-layout compatibility

KV cache
  -> an independent precision and capacity decision
```

These patches were necessary for checkpoint compatibility and backend execution in our environment, before any decision to quantize the KV cache. A reproduction should preserve the complete working runtime and validate each change against that baseline.

With those later integration changes distinguished from the initial baseline, the earlier memory-tuning sequence is easier to read on its own terms.

Once the checkpoint loaded, memory became the main constraint. The initial goal was a context window close to 240,000 tokens with a relatively generous sequence limit. Our first configuration requested 245,760 tokens, used `gpu-memory-utilization=0.948`, and allowed up to 64 sequences.

It failed the KV-cache capacity check. The runtime required 6.85 GiB and had 6.79 GiB available. Its calculated maximum was 243,936 tokens, just below the requested context length.

That small difference encouraged a tempting adjustment: increase the memory-utilization setting. We tried 0.96. In this environment, that left too little physical headroom for other work during initialization, and problems appeared around autotuning and CUDA graphs. The setting increased the budget available to the runtime, but it did not create additional memory on the device.

Reducing the requested context to 240,000 while returning utilization to 0.948 got us further. The runtime reported KV capacity for 242,909 tokens, completed startup, and answered its health check.

Then the first real inference failed. The failing operation wanted another 192 MiB, but only 32.44 MiB was free.

At that point the distinction between startup success and inference success was concrete. The server had enough memory for the checks it had already performed. It did not have enough for an allocation reached during generation.

The full progression looked like this:

| Maximum context | Memory utilization | Maximum sequences | Observed result |
| ---: | ---: | ---: | --- |
| 245,760 | 0.948 | 64 | Failed KV-capacity validation: 6.85 GiB required, 6.79 GiB available |
| 245,760 | 0.96 | 64 | Insufficient headroom around autotuning and CUDA graphs |
| 240,000 | 0.948 | 64 | Started successfully, then failed during generation |
| 200,000 | 0.94 | 32 | With expandable segments enabled, failed during startup |
| 200,000 | 0.94 | 32 | Without that allocator setting, passed sequential requests but failed under concurrent traffic |
| 131,072 | 0.92 | 16 | Completed three rounds of 16 concurrent requests |

The inference failure pointed into the GDN linear-attention path. The trace identified a temporary state-tensor allocation during forward execution. That allocation needed free memory beyond the resources already accounted for by the successful startup and KV-capacity checks.

The GPU was also shared with other workloads. Its nominal capacity was therefore not the same as the budget available to this model. The amount of free memory at a critical allocation depended on more than the model weights and configured KV cache.

Our working explanation was that the deployment had too little margin for temporary allocations during generation, and the concurrent workload exposed that margin more reliably than isolated requests. The failure locations and reported allocation sizes supported that explanation. We did not collect a full allocation-level memory profile, so we could not quantify every contributor to the peak.

The next configuration was particularly misleading. At 200,000 tokens of context, utilization of 0.94, and a sequence limit of 32, several sequential requests succeeded. Requests through the API gateway also completed. If the acceptance criterion had been “the endpoint answers,” we could have stopped there.

Concurrent traffic produced another OOM. This time the failing allocation requested 290 MiB with only 150 MiB free. The container restarted, making it clear that the earlier successful responses had not exercised the workload that mattered.

That result changed the way we evaluated subsequent attempts. A successful sequential request remained useful as a functional check, but it could no longer stand in for a concurrency test. We needed to send overlapping requests and observe the process while generation was in progress.

We also tried the allocator option `expandable_segments`. In this setup, it introduced a different failure mode: an OOM during memory mapping, followed by a `pidfd_getfd: Operation not permitted` error in the PLE offload worker. We removed the setting.

That experiment did not establish that expandable segments are generally unsuitable for inference. It established that the option did not solve this deployment's problem and made this particular startup path fail. Carrying it forward would have added another unresolved variable.

The configuration that passed our concurrent test reduced three limits together: maximum context, maximum sequences, and memory utilization. The resulting settings were:

```text
max-model-len: 131072
kv-cache-dtype: bfloat16
gpu-memory-utilization: 0.92
max-num-seqs: 16
PLE CPU offload: enabled
MTP speculative tokens: 1
prefix caching: enabled
```

The runtime also used the Qwen reasoning and tool-call parsers. Multimodal request limits were configured, but the test described below used text requests; it did not validate image or video workloads.

Changing three memory-related parameters at once means we cannot attribute the success to one of them in isolation. We did not perform an exhaustive sweep that held every other factor constant. The objective at this stage was to find a usable baseline in the actual environment. The combination above supplied that baseline for the workload we ran.

We tested it with three rounds of 16 concurrently submitted chat-completion requests. Each request asked for a roughly 300-word essay about distributed systems. The generation limit was 700 tokens, temperature was 0.8, and the client timeout was 180 seconds. Each round finished before the next one began.

The script counted responses containing a `finish_reason` field and inspected the container restart count. It also sampled GPU memory approximately every two seconds during the beginning of each round.

The recorded results were:

| Round | Responses with a finish reason | Recorded container restarts |
| --- | ---: | ---: |
| 1 | 16 / 16 | 0 |
| 2 | 16 / 16 | 0 |
| 3 | 16 / 16 | 0 |
| Total | 48 / 48 | 0 |

The highest sampled GPU memory usage was **92,712 MiB**, approximately **90.54 GiB**. The final memory snapshot reported **4,537 MiB free**, approximately **4.43 GiB**.

Those numbers need a little care. The samples describe device-wide usage, including other processes on the GPU. Sampling at intervals can miss short peaks, and the script did not continuously sample for the entire duration of every request. The free-memory reading was a final snapshot; it should not be presented as a continuously guaranteed reserve under load.

The completion count also has a specific meaning. All 48 responses passed the script's check for the presence of a finish reason. That check did not evaluate the quality of the essays or require that every response ended naturally rather than reaching its token limit. We did not derive a throughput claim, a latency percentile, or a quality score from this test.

Most importantly, a server configured for 131,072 tokens of context was not being tested with 131,072-token prompts. These were short prompts with bounded generation. The result established that the selected configuration handled this concurrent text workload without a recorded restart. Testing long-context prefill, mixed prompt lengths, sustained traffic, and multimodal requests would require additional runs.

After the concurrent test, requests through the API gateway also completed. That checked a separate part of the deployment: the model was reachable through the intended serving route after the container changes. It complemented the direct inference test, while leaving the same workload limits in place.

What made this exercise useful was the sequence of increasingly meaningful checks. Download completion established that the artifacts were present. Successful loading established that the runtime could interpret them. A health response established that the service was up. A generated response exercised inference. Concurrent requests exposed a memory problem that the earlier checks had missed.

The same distinction applied to the fixes. The loader patch addressed a checkpoint-layout mismatch. The configuration override got past a runtime compatibility issue but still needed separate quality validation. The final memory settings addressed the observed failures under a particular workload. None of those changes, on its own, justified calling every aspect of the deployment validated.

We ended with a smaller context target than we initially wanted, a lower sequence limit, and a configuration that completed the chosen concurrency check. That was a much more useful foundation for further experiments than a larger advertised context attached to a process that restarted when traffic overlapped.

Further experiments need separate baselines for the runtime and the representation being changed. Long-context and sustained-load tests should exercise the complete custom stack. Any KV-cache quantization experiment should retain the established backend integration and be evaluated independently. Keeping those changes distinguishable lets us measure memory consumption, reliability, and answer quality without attributing a result to the wrong patch.

