# Oryndral

Oryndral is a Rust inference engine for running supported GGUF language models locally.

## Build and try it

Use stable Rust and a C linker. On Windows, install Visual Studio Build Tools with the C++ workload and a Windows SDK. On Linux, install the standard build tools.

From the repository folder, build the engine and download the 19 MB TinyStories example. These commands use Windows PowerShell. On Linux, use `mkdir -p models` and `curl` in the download step.

```powershell
cargo build --release --locked
New-Item -ItemType Directory -Force models | Out-Null
curl.exe -fL -o models/stories15M-q4_0.gguf https://huggingface.co/ggml-org/models-moved/resolve/499bc8821c6b12b4e53c5bffcb21ec206f212d81/tinyllamas/stories15M-q4_0.gguf
cargo run --release --locked -- run models/stories15M-q4_0.gguf --prompt "Once upon a time" --temperature 0 --max-tokens 24 --threads 4 --context 128
```

The example continues the prompt with a short story. The [benchmark record](docs/benchmark.md) lists the download's SHA-256.

To use the same model in the browser or with an API client, start the server.

```text
cargo run --release --locked -- serve models/stories15M-q4_0.gguf --model-id stories --bind 127.0.0.1:8080 --context 128
```

Open [Oryndral chat](http://127.0.0.1:8080/) or point an OpenAI-compatible client at `http://127.0.0.1:8080/v1` with model ID `stories`. For this base model, use a single prompt.

## What works

- Single-file GGUF inference for supported Llama and dense Qwen2 layouts, including Mythos-nano.
- CPU inference, dense layer streaming, and expert streaming for supported Llama MoE models. NVIDIA layer offload is available with a CUDA build. [Architecture](docs/architecture.md) · [Verified models and benchmarks](docs/benchmark.md)
- Local browser chat, an OpenAI-compatible text and chat API, and Hugging Face downloads via `serve --hf`. The chat keeps recognized reasoning blocks separate from the answer.
- A [Windows desktop app](desktop/README.md) that starts the engine and opens the chat.

## License

The engine and desktop app are available under the [MIT License](LICENSE). Model weights are downloaded separately and have their own licenses.
