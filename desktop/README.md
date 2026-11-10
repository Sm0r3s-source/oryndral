# Oryndral Desktop for Windows

Open **Oryndral Desktop.exe**, click **Choose model…** to pick a single-file GGUF, select **CPU** or **NVIDIA GPU**, then click **Start server**. The Oryndral chat opens in the desktop window. **Copy API address** provides the loopback URL for other OpenAI-compatible clients. **Stop server** and closing the window stop the engine started by this instance. If the engine fails to load a model, its error appears in the main pane and in **Server log**.

The window follows the Windows light or dark app theme and uses the same colors as the chat page. Links in a reply open in the default browser.

The app uses the Microsoft Edge WebView2 Runtime already installed on supported Windows systems. It binds the engine to loopback port 8177, so the API address and the chat's saved settings stay the same between runs. If that port is in use, the engine takes any free port. No external service is needed for a model already on disk.

## Build and package

Run `build.ps1` with PowerShell and Visual Studio 2022 Build Tools installed. The script locates Roslyn and the local WebView2 SDK assemblies. Use `-Compiler` and `-WebView2Directory` for an alternate Roslyn installation and WebView2 SDK directory. The latter must contain the SDK's `Microsoft.Web.WebView2.Core.dll` and `Microsoft.Web.WebView2.Wpf.dll`. Use `-WebView2Loader` if the matching x64 loader DLL is in a separate directory.

Keep the three WebView2 SDK DLLs alongside the desktop executable. To package the Rust engines, use this layout.

```text
Oryndral Desktop.exe
Microsoft.Web.WebView2.Core.dll
Microsoft.Web.WebView2.Wpf.dll
WebView2Loader.dll
engine/
  cpu/oryndral.exe
  cuda/oryndral.exe
models/
```

The CUDA build expects its CUDA 12 runtime DLLs in `engine/cuda/` or `cuda-runtime/`. Supply the CPU build for machines without NVIDIA CUDA. Model weights can stay anywhere on disk. In the development workspace the desktop host also recognizes the engine and model locations under `work/`.

`Oryndral.svg` is the editable icon artwork. The desktop build embeds the matching multi-size Windows icon from `Oryndral.ico`.

WebView2 SDK files and NVIDIA CUDA runtime files retain their respective distribution terms. [WebView2 distribution instructions](https://learn.microsoft.com/microsoft-edge/webview2/concepts/distribution) · [NVIDIA CUDA licensing](https://docs.nvidia.com/cuda/eula/index.html)
