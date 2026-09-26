# Bundle validation

- Four CPU wrapper tests passed: invalid routes are zero-weighted, valid routes are preserved, compatible wrappers share buffers, and a different token capacity receives separate buffers.
- The PLE diff passed `git apply --check` against the saved base module.
- Applying the diff reproduced the recorded patched module after line-ending normalization. Windows newline differences prevent a raw byte-for-byte comparison of those local copies.
- Public text was scanned for known internal identifiers, private paths, dated infrastructure labels, and common token patterns. Raw logs, model configuration, container inspect output, and sessions were not included.
- No live model, GPU kernel, container image build, performance test, or numerical-equivalence test was run while assembling this bundle.

The historical measurements are documented separately from these packaging checks.
