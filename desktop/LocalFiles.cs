using System;
using System.IO;

internal sealed class LocalFiles
{
    private readonly string directory = AppDomain.CurrentDomain.BaseDirectory;
    private readonly string work;

    public string CpuEngine => ExistingFile(Path.Combine(directory, "engine", "cpu", "oryndral.exe"),
        AtWork("target", "release", "oryndral.exe"));
    public string CudaEngine => ExistingFile(Path.Combine(directory, "engine", "cuda", "oryndral.exe"),
        AtWork("target-cuda", "release", "oryndral.exe"));
    public string CudaBin => ExistingCudaDirectory(Path.Combine(directory, "engine", "cuda"),
        Path.Combine(directory, "cuda-runtime"), AtWork("cuda-12.9.1", "toolkit", "bin"));
    public bool CudaReady => CudaEngine != null && CudaBin != null
        && File.Exists(Path.Combine(Environment.SystemDirectory, "nvcuda.dll"));
    public string SuggestedModel => ExistingFile(
        Path.Combine(directory, "models", "mythos-nano-Q4_K_M.gguf"),
        AtWork("mythos-nano-Q4_K_M.gguf"),
        Path.Combine(directory, "models", "stories15M-q4_0.gguf"),
        AtWork("stories15M-q4_0.gguf"));
    public string BrowserDataDirectory => AtWork("desktop-webview")
        ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Oryndral", "DesktopWebView");

    public LocalFiles()
    {
        for (DirectoryInfo current = new DirectoryInfo(directory); current != null; current = current.Parent)
        {
            if (!File.Exists(Path.Combine(current.FullName, "outputs", "oryndral", "Cargo.toml")))
                continue;
            work = Path.Combine(current.FullName, "work");
            break;
        }
    }

    private string AtWork(params string[] parts)
    {
        return work == null ? null : Path.Combine(work, Path.Combine(parts));
    }

    private static string ExistingFile(params string[] options)
    {
        foreach (var path in options)
            if (File.Exists(path))
                return path;
        return null;
    }

    private static string ExistingCudaDirectory(params string[] options)
    {
        foreach (var path in options)
            if (path != null && File.Exists(Path.Combine(path, "cublas64_12.dll"))
                && File.Exists(Path.Combine(path, "cudart64_12.dll"))
                && File.Exists(Path.Combine(path, "curand64_10.dll"))
                && File.Exists(Path.Combine(path, "nvrtc64_120_0.dll")))
                return path;
        return null;
    }
}
