# Recorded baseline results

The initial baseline used bfloat16 KV cache, maximum model length 131072, memory utilization 0.92, and a maximum of 16 sequences. These measurements belong to that initial run, not a benchmark of the later B12x integration.

Three successive rounds submitted 16 requests concurrently. Each requested an approximately 300-word essay about distributed systems, with `max_tokens=700`, temperature 0.8, and a 180-second client timeout.

| Round | Responses containing a finish reason | Recorded restarts |
| --- | ---: | ---: |
| 1 | 16/16 | 0 |
| 2 | 16/16 | 0 |
| 3 | 16/16 | 0 |

The highest sampled device memory usage was 92712 MiB. The final free-memory snapshot was 4537 MiB. Sampling occurred approximately every two seconds at the beginning of each round, not continuously for the full test. Values include other GPU processes.

The check counted the presence of a finish reason, not semantic quality or necessarily a natural stop. It did not validate a filled 128K context, sustained service reliability, multimodal workloads, or throughput. Private raw sessions are excluded from this public bundle.

The B12x workspace estimate of about 18.75 GiB comes from approximately 400 MiB across 48 layers. It is an estimate of duplicated allocations, not a measured before-and-after saving in the 48-request test.
