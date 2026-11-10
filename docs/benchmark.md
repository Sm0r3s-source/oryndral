# Reproducing the CPU benchmarks

The test host ran Windows 11 build 26200 with an Intel Core i5-12450HX (12 logical processors, AVX2) and 16 GiB installed RAM. The fixed-prompt TinyStories tests used four CPU threads, greedy decoding, the five-token prompt `Once upon a time` including BOS, and a 24-token generation limit. Each reported median uses five fresh processes after one warm-up run. The OS file cache remained warm. Page residency was unmeasured.

## Fixtures

From the project directory, obtain the exact files with PowerShell. On Linux, substitute `curl` for `curl.exe` and use `sha256sum` to check the downloads.

```powershell
curl.exe -fL -o stories15M-q4_0.gguf https://huggingface.co/ggml-org/models-moved/resolve/499bc8821c6b12b4e53c5bffcb21ec206f212d81/tinyllamas/stories15M-q4_0.gguf
curl.exe -fL -o stories15M_MOE-Q8_0.gguf https://huggingface.co/ggml-org/stories15M_MOE/resolve/b6dd737497465570b5f5e962dbc9d9454ed1e0eb/stories15M_MOE-Q8_0.gguf
Get-FileHash stories15M*.gguf -Algorithm SHA256
```

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `stories15M-q4_0.gguf` | 19,077,344 | `66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739` |
| `stories15M_MOE-Q8_0.gguf` | 39,390,272 | `c7aa6863f9a4b3cdf19716e2c95622dcbd3bd06989324bf1ac8e60486ef8e881` |

Both fixtures use MIT-licensed weights. The dense checkpoint is a GGUF conversion of Karpathy's TinyStories Llama model. [TinyLlamas model card](https://huggingface.co/karpathy/tinyllamas), [dense GGUF revision](https://huggingface.co/ggml-org/models-moved/commit/499bc8821c6b12b4e53c5bffcb21ec206f212d81), [MoE GGUF revision](https://huggingface.co/ggml-org/stories15M_MOE/commit/b6dd737497465570b5f5e962dbc9d9454ed1e0eb).

## Oryndral

```powershell
cargo run --locked --release -- benchmark stories15M-q4_0.gguf --prompt 'Once upon a time' --temperature 0 --top-p 1 --seed 42 --max-tokens 24 --threads 4 --context 128 --mode resident
cargo run --locked --release -- benchmark stories15M-q4_0.gguf --prompt 'Once upon a time' --temperature 0 --top-p 1 --seed 42 --max-tokens 24 --threads 4 --context 128 --mode dense-stream
foreach ($cache in @(0, 8, 64)) {
    cargo run --locked --release -- benchmark stories15M_MOE-Q8_0.gguf --prompt 'Once upon a time' --temperature 0 --top-p 1 --seed 42 --max-tokens 24 --threads 4 --context 128 --mode moe-stream --expert-cache-mib $cache
}
```

Warm each case once, then run five measured rounds, rotating case order in each round. Each run starts a fresh process and empty expert cache. Process peak RSS is the process high-water mark. Logical model reads include cached reads. Physical storage bytes require an operating system counter that attributes disk reads to the process. CPU runs have no GPU model allocation.

| Dense mode | Prefill tokens/s | Decode tokens/s | Time to first token after load | Peak working set | Model bytes read during generation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Resident | 1,669.39 | 724.76 | 3.16 ms | 40.70 MiB | 0 |
| Dense streaming | 755.14 | 241.31 | 6.87 ms | 40.69 MiB | 80,953,344 |

All ten measured runs emitted the same dense reference text and 24 token IDs. Decode ranged from 652.81 to 929.65 tokens/s in resident mode and from 217.57 to 247.31 tokens/s in dense streaming. Median load time was 195.80 ms and 197.56 ms respectively. The planner estimated 30,829,184 bytes for resident mode and 28,018,304 for dense streaming. The measured process peaks differed by 4,096 bytes, within the spread across runs.

With `--ram-mib 27`, the resident command rejects the estimated placement and dense streaming emits the same text. This limit applies to the planner's estimate. The process high-water mark also includes runtime overhead.

| MoE LRU capacity | Prefill tokens/s | Decode tokens/s | Time to first token after load | Peak working set | Model bytes read during generation | Expert hit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 MiB | 142.98 | 127.11 | 35.17 ms | 45.09 MiB | 236,888,064 | 0.00% |
| 8 MiB | 417.49 | 236.40 | 12.09 ms | 51.28 MiB | 190,356,480 | 19.64% |
| 64 MiB | 340.57 | 660.26 | 14.88 ms | 61.93 MiB | 16,920,576 | 92.86% |

The cache cases had 0, 66, and 312 hits out of 336 expert lookups. All three capacities reproduced the llama.cpp server's 24 sampled token IDs and continuation text.

An equal-budget rerun compared the default LRU against LFU at 8 MiB, alternating policies for five measured runs each after one warm-up. Both policies reproduced all 24 reference IDs in every run. LFU uses per-entry access counts and evicts the oldest entry when counts tie.

| Policy at 8 MiB | Decode tokens/s | Expert hits / requests | Hit rate | Logical GGUF bytes read |
| --- | ---: | ---: | ---: | ---: |
| LRU | 257.74 | 66 / 336 | 19.64% | 190,356,480 |
| LFU | 369.74 | 159 / 336 | 47.32% | 124,789,248 |

Reproduce by setting `--expert-cache-mib 8` in the MoE command above and alternating `--expert-cache-policy lru` and `--expert-cache-policy lfu`. The measured speedup applies to this fixture and prompt.

## llama.cpp comparison

Download and unpack the Windows x64 CPU package from llama.cpp release `b12230` (`a4d4f2fd4`). The ZIP used here had SHA-256 `a787fb94d48e6d93aa5809133f5f984360a5713b20d2f9ee1ccc4b22cd9cce56`. Set `$llama` to the extracted directory. [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b12230).

```powershell
$llama = '.\llama-b12230'
foreach ($model in @('stories15M-q4_0.gguf', 'stories15M_MOE-Q8_0.gguf')) {
    foreach ($run in 0..5) {
        & "$llama\llama-completion.exe" -m $model -p 'Once upon a time' -n 24 -t 4 -tb 4 -c 128 -ngl 0 -fa off -ctk f32 -ctv f32 --temp 0 --seed 42 --repeat-penalty 1 --top-k 0 --top-p 1 --min-p 0 -lm mmap --no-conversation --simple-io --no-display-prompt --perf --verbose-prompt > "$model.$run.txt" 2> "$model.$run.log"
    }
}
```

llama.cpp padded the requested context from 128 to 256 internally. `--perf` measures five prompt evaluations and 23 incremental decode evaluations because prefill supplies the logits for the first of the 24 generated tokens. Oryndral's 23-step decode interval also includes sampling and text decoding. The process timings below sample peak Windows working set with `K32GetProcessMemoryInfo` and measure first stdout byte from process launch, including model initialization. Oryndral's first-token timing starts after model load.

| llama.cpp fixture | Prompt tokens/s | Decode steps/s | Peak working set | Launch to first byte |
| --- | ---: | ---: | ---: | ---: |
| Stories15M Q4_0 | 4,608.29 | 1,566.22 | 38.90 MiB | 89.02 ms |
| Stories15M MoE Q8_0 | 1,410.44 | 1,132.67 | 50.76 MiB | 95.43 ms |

The measured decode ranges were 1,532.01–1,681.04 and 972.52–1,322.90 steps/s respectively. Five-token prompt timings are sensitive to timer and scheduling variation. The prompt rate ranged from 1,724.14 to 4,789.27 tokens/s for the dense fixture and 1,278.12 to 1,882.53 for MoE.

The independent synthetic benchmark below used F16 KV because this `llama-bench` build rejected F32. It measured a mean 6,590.15 ± 572.07 prompt tokens/s for `pp5` and 2,017.46 ± 109.14 decode steps/s for `tg24` on the dense fixture. These synthetic runs use separate prompt and generation cases.

```powershell
& "$llama\llama-bench.exe" -m stories15M-q4_0.gguf -p 5 -n 24 -t 4 -ngl 0 -fa off -ctk f16 -ctv f16 -lm mmap -r 5 -o json
```

## Greedy output reference

The reference continuations below were stable in all six llama.cpp CLI runs. The llama.cpp b12230 server's `/completion` endpoint also returned the actual sampled IDs, which matched Oryndral on both fixtures. The five prompt IDs, including BOS, are `[1, 9038, 2501, 263, 931]`.

Run the reference server with each fixture in turn, then make the request in a separate terminal.

```powershell
& "$llama\llama-server.exe" -m stories15M-q4_0.gguf -c 128 -t 4 -tb 4 -ngl 0 -fa off -ctk f32 -ctv f32 --host 127.0.0.1 --port 8081
```

```powershell
$body = @{prompt = 'Once upon a time'; n_predict = 24; temperature = 0; top_k = 0; top_p = 1; min_p = 0; repeat_penalty = 1; seed = 42; return_tokens = $true} | ConvertTo-Json
(Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8081/completion -ContentType 'application/json' -Body $body).tokens
```

| Fixture | 24-token continuation |
| --- | --- |
| Dense | `, there was a little girl named Lily. She loved to play outside and explore the world around her. One day` |
| MoE | `, there was a little girl named Lily. She had a big sister named Rose. Lily was very pretty and` |

```text
Dense: 29892, 727, 471, 263, 2217, 7826, 4257, 365, 2354, 29889, 2296, 18012, 304, 1708, 5377, 322, 26987, 278, 3186, 2820, 902, 29889, 3118, 2462
MoE:   29892, 727, 471, 263, 2217, 7826, 4257, 365, 2354, 29889, 2296, 750, 263, 4802, 9883, 4257, 14008, 29889, 365, 2354, 471, 1407, 5051, 322
```

These small fixtures fit in RAM. The measurements cover CPU correctness and a warm-filesystem baseline. Cold physical storage and CUDA require separate measurements.

## SmolLM2 under a Windows process limit

SmolLM2-135M-Instruct Q4_K_M is a 105,454,144-byte GGUF with grouped-query attention and an embedded ChatML template. Oryndral used the upstream tokenizer JSON. Both placement modes emitted `The capital of France is Paris.` with sampled IDs `[504, 3575, 282, 4649, 314, 7042, 30, 2]`, including EOS. The llama.cpp b12230 server returned the same IDs when given the rendered 39-token prompt from Oryndral's benchmark JSON. [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) · [GGUF conversion](https://huggingface.co/QuantFactory/SmolLM2-135M-Instruct-GGUF/tree/476854d00ede130660aba430d15f9347ad2e7d0e)

```powershell
curl.exe -fL -o smollm2-135m-q4_k_m.gguf https://huggingface.co/QuantFactory/SmolLM2-135M-Instruct-GGUF/resolve/476854d00ede130660aba430d15f9347ad2e7d0e/SmolLM2-135M-Instruct.Q4_K_M.gguf
curl.exe -fL -o smollm2-tokenizer.json https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/12fd25f77366fa6b3b4b768ec3050bf629380bac/tokenizer.json
Get-FileHash smollm2-135m-q4_k_m.gguf,smollm2-tokenizer.json -Algorithm SHA256

cargo build --release --locked
$ory = '.\target\release\oryndral.exe'
$argsSmol = @('benchmark', 'smollm2-135m-q4_k_m.gguf', '--tokenizer', 'smollm2-tokenizer.json', '--chat', '--prompt', 'Name the capital of France in one word.', '--context', '128', '--threads', '4', '--max-tokens', '12', '--temperature', '0')
foreach ($mode in @('resident', 'dense-stream')) {
    foreach ($run in 0..5) {
        & $ory @argsSmol --mode $mode > "smol-$mode-$run.json"
    }
}
```

| File | SHA-256 |
| --- | --- |
| GGUF | `8030f04528538d47bda434f6f0bdf3952c40a58123e4d5e755332f23731a8684` |
| tokenizer.json | `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` |

The following rows are medians of five measured runs per mode after a warm-up, using warm filesystem pages and the same Windows CPU.

| Mode | Prefill tokens/s | Decode tokens/s | Time to first token after load | Peak working set | Logical GGUF bytes read during generation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Resident | 51.70 | 45.40 | 754.8 ms | 151.89 MiB | 0 |
| Dense streaming | 38.97 | 11.30 | 1,001.2 ms | 78.52 MiB | 735,851,520 |

Separate hard-limit checks used a Windows Job Object to cap process commit. Oryndral resident mode hit the allocation limit at 96 MiB. With the same 39-token rendered prompt, requested context, 16-token prefill batch and F32 KV, Oryndral dense streaming completed at both limits. The tested llama.cpp build used mmap with repacking disabled. It could initialize the context and produce the same text at 96 MiB and failed to allocate its compute buffers at 80 MiB. llama.cpp decoded much faster in its successful run. The operating system had enough physical memory to cache model pages, so this measures constrained process memory with warm filesystem pages.

| Process commit limit | Oryndral dense streaming | llama.cpp mmap, batch 16, repacking disabled |
| --- | --- | --- |
| 80 MiB | Completed at 75.43 MiB peak commit | Context allocation failed |
| 96 MiB | Completed at 76.46 MiB peak commit | Completed at 83.89 MiB peak commit |

```powershell
python -m pip install pywin32
python scripts/windows_memory_cap.py --mib 96 -- $ory @argsSmol --mode resident
$rendered = (Get-Content smol-dense-stream-0.json -Raw | ConvertFrom-Json).prompt
foreach ($limit in @(80, 96)) {
    python scripts/windows_memory_cap.py --mib $limit -- $ory @argsSmol --mode dense-stream
    python scripts/windows_memory_cap.py --mib $limit -- "$llama\llama-completion.exe" -m smollm2-135m-q4_k_m.gguf -p $rendered -n 12 -t 4 -tb 4 -c 128 -b 16 -ub 16 -ngl 0 -fa off -ctk f32 -ctv f32 --temp 0 -lm mmap --no-repack --no-conversation --simple-io --no-display-prompt --perf
}
```

## mistral.rs and Colibrì

The official mistral.rs v0.9.5 Windows CPU archive had SHA-256 `64d567c299a5c97ef265aa0362649aafb968af47706810df640f74fe8ea4702a`. Its stock GGUF loader declined `stories15M-q4_0.gguf` because the file omits `llama.attention.head_count_kv`. An external model config supplied the six KV heads and kept the GGUF bytes identical. The MoE fixture loaded directly and SmolLM2 used its matching `tokenizer.json`. [Release archive and checksum](https://github.com/EricLBuehler/mistral.rs/releases/expanded_assets/v0.9.5) · [Benchmark command reference](https://docs.mistralrs.dev/reference/cli/bench/)

| Fixture | Prompt tokens/s | TTFT | Decode tokens/s | Time per output token |
| --- | ---: | ---: | ---: | ---: |
| Stories15M Q4_0 with model config | 1,525.9 ± 189.0 | 3.33 ms | 720.7 ± 32.9 | 1.39 ms |
| Stories15M MoE Q8_0 | 138.0 ± 4.6 | 36.26 ms | 82.2 ± 1.1 | 12.16 ms |
| SmolLM2-135M-Instruct Q4_K_M | 53.1 ± 1.4 | 94.30 ms | 45.9 ± 2.3 | 21.86 ms |

The built-in benchmark uses synthetic token workloads and measures prompt latency and decode in separate cases. Its decode case uses 24 tokens at depth five. The fixed-prompt timings above use sampled text and different timing boundaries.

```powershell
$mistral = 'C:\path\to\mistralrs-cpu'
$env:RAYON_NUM_THREADS = '4'
$benchArgs = @('--cpu','--max-model-len','128','--prompt-len','5','--depth','5','--gen-len','24','--iterations','5','--warmup','1')
& "$mistral\mistralrs.exe" bench @benchArgs -f (Resolve-Path stories15M_MOE-Q8_0.gguf).Path
& "$mistral\mistralrs.exe" bench @benchArgs -f (Resolve-Path smollm2-135m-q4_k_m.gguf).Path --tokenizer (Resolve-Path smollm2-tokenizer.json).Path
```

For the dense fixture, create `stories15m-config/config.json` with the values below and pass its directory with `--tok-model-id`.

```json
{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 288,
  "intermediate_size": 768,
  "num_hidden_layers": 6,
  "num_attention_heads": 6,
  "num_key_value_heads": 6,
  "vocab_size": 32000,
  "max_position_embeddings": 128,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "rms_norm_eps": 0.00001,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false
}
```

```powershell
& "$mistral\mistralrs.exe" bench @benchArgs -f (Resolve-Path stories15M-q4_0.gguf).Path --tok-model-id (Resolve-Path stories15m-config).Path
```

Colibrì's GGUF importer requires a supported architecture, and its published list excludes `llama`, which is the architecture tag on both TinyStories fixtures. A same-checkpoint comparison needs an overlapping architecture such as OLMoE. [Colibrì GGUF architecture list](https://github.com/JustVugg/colibri/blob/main/docs/gguf.md#supported-architectures)

## Mythos-nano Qwen2 compatibility

The CPU compatibility check used the publisher's Mythos-nano Q4_K_M GGUF, 1,929,902,176 bytes with SHA-256 `8eefaab7ef34b2e8ab046f65905cbf4d11fc6f8533df99d839a0734467b9b188`. Oryndral rendered the embedded chat template and matched all 30 prompt IDs and all 24 greedy sample IDs from llama.cpp b12230 for `What is 2 + 2? Answer briefly.`. Both engines used four CPU threads and a requested context of 512. [Pinned GGUF revision](https://huggingface.co/squ11z1/Mythos-nano-GGUF/commit/388d2fc2f04561dac247c6e6bfd699c8ade66e98)

```powershell
curl.exe -fL -o mythos-nano-Q4_K_M.gguf https://huggingface.co/squ11z1/Mythos-nano-GGUF/resolve/388d2fc2f04561dac247c6e6bfd699c8ade66e98/mythos-nano-Q4_K_M.gguf
Get-FileHash mythos-nano-Q4_K_M.gguf -Algorithm SHA256
$env:ORYNDRAL_TEST_MYTHOS_GGUF = (Resolve-Path mythos-nano-Q4_K_M.gguf).Path
cargo test --release --locked --test inference mythos_nano_matches_llama_cpp -- --ignored
```

To reproduce the reference, run the benchmark command below and pass its rendered `prompt` to the same llama.cpp `/completion` endpoint used above, setting `n_predict` to 24, `temperature` to zero and `return_tokens` to true. Start the reference server with `-c 512 -b 16 -ub 16 -t 4 -tb 4 -ngl 0 -fa off -ctk f32 -ctv f32`. Compare those IDs with `generation.completion_token_ids`.

```text
cargo run --release --locked -- benchmark mythos-nano-Q4_K_M.gguf --chat --prompt "What is 2 + 2? Answer briefly." --context 512 --threads 4 --temperature 0 --max-tokens 24
```

## Windows CUDA verification

Oryndral built and ran on the RTX 4050 Laptop GPU (6 GiB, compute capability 8.9) using CUDA 12.9.1, driver 621.01 and the Windows x64 MSVC toolchain. The CUDA release build used NVIDIA's component archives verified against the official SHA-256 values. MSVC emitted an LNK4098 runtime-library warning and linked the executable successfully. [NVIDIA CUDA 12.9.1 component manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json)

The server loaded all 36 Mythos-nano transformer blocks on CUDA with a configured context of 4,096 and returned a chat completion over HTTP. The median of three GPU-wide memory samples rose from 670 MiB before launch to 2,352 MiB after loading, an increase of 1,682 MiB. The embedding and output weights stay on CPU.

The separate 512-context, 24-token CLI run used the same prompt and sampling as the CPU comparison above. CUDA matched the first 21 greedy token IDs and then chose a different continuation. This single run measured 45.3 decode tokens/s. With the output limit raised to 192, the same CUDA configuration answered `4` after its reasoning and emitted a stop token at step 72.

```text
cargo run --release --locked --features cuda -- benchmark mythos-nano-Q4_K_M.gguf --mode hybrid --gpu-layers 36 --chat --prompt "What is 2 + 2? Answer briefly." --context 512 --threads 4 --temperature 0 --max-tokens 24
cargo run --release --locked --features cuda -- serve mythos-nano-Q4_K_M.gguf --mode hybrid --gpu-layers 36 --model-id mythos-nano --context 4096
```
