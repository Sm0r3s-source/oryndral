using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Win32.SafeHandles;

internal sealed class ManagedServer : IDisposable
{
    private static readonly Regex ListenLine = new Regex(
        @"^Oryndral serving .+ at http://127\.0\.0\.1:([0-9]{1,5})/v1$",
        RegexOptions.CultureInvariant);

    private readonly Process process;
    private readonly ChildJob job;
    private readonly TaskCompletionSource<Uri> listening =
        new TaskCompletionSource<Uri>(TaskCreationOptions.RunContinuationsAsynchronously);
    private int disposed;

    public event Action<string> Output;
    public event Action<ManagedServer> Exited;

    public ManagedServer(string engine, string model, string cudaBin, int? context)
    {
        string arguments = "serve " + QuoteArgument(model) + " --bind 127.0.0.1:0";
        if (cudaBin != null)
            arguments += " --mode hybrid";
        if (context.HasValue)
            arguments += " --context " + context.Value;

        var start = new ProcessStartInfo
        {
            FileName = engine,
            Arguments = arguments,
            WorkingDirectory = Path.GetDirectoryName(engine),
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardErrorEncoding = Encoding.UTF8,
            StandardOutputEncoding = Encoding.UTF8,
        };
        if (!String.IsNullOrEmpty(cudaBin))
            start.EnvironmentVariables["PATH"] = cudaBin + ";" + start.EnvironmentVariables["PATH"];

        process = new Process { StartInfo = start };
        process.OutputDataReceived += ReadOutput;
        process.ErrorDataReceived += ReadOutput;
        process.Exited += ProcessExited;
        job = ChildJob.Create();
    }

    public async Task<Uri> StartAsync(CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();
        if (!process.Start())
            throw new InvalidOperationException("The Oryndral engine could not start.");
        job?.Assign(process);
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        process.EnableRaisingEvents = true;

        using (cancellation.Register(() => listening.TrySetCanceled()))
        {
            var address = await listening.Task;
            await WaitForHealthAsync(address, cancellation);
            return address;
        }
    }

    private static async Task WaitForHealthAsync(Uri address, CancellationToken cancellation)
    {
        using (var handler = new HttpClientHandler { UseProxy = false, AllowAutoRedirect = false })
        using (var client = new HttpClient(handler) { Timeout = TimeSpan.FromSeconds(2) })
        {
            for (int attempt = 0; attempt < 12; attempt++)
            {
                try
                {
                    using (var response = await client.GetAsync(new Uri(address, "health"), cancellation))
                        if (response.IsSuccessStatusCode)
                            return;
                }
                catch (HttpRequestException) { }
                catch (TaskCanceledException) when (!cancellation.IsCancellationRequested) { }
                await Task.Delay(250, cancellation);
            }
        }
        throw new InvalidOperationException("The engine opened a local port but did not become ready.");
    }

    private void ReadOutput(object sender, DataReceivedEventArgs args)
    {
        if (String.IsNullOrWhiteSpace(args.Data))
            return;
        Output?.Invoke(args.Data);
        var match = ListenLine.Match(args.Data);
        int port;
        if (match.Success && Int32.TryParse(match.Groups[1].Value, out port)
            && port > 0 && port <= 65535)
            listening.TrySetResult(new Uri("http://127.0.0.1:" + port + "/"));
    }

    private void ProcessExited(object sender, EventArgs args)
    {
        if (Volatile.Read(ref disposed) != 0)
            return;
        try { process.WaitForExit(); }
        catch (InvalidOperationException) { return; }
        listening.TrySetException(new InvalidOperationException(
            "The Oryndral engine exited. Open server details for the error."));
        Exited?.Invoke(this);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposed, 1) != 0)
            return;
        process.Exited -= ProcessExited;
        listening.TrySetCanceled();
        job?.Dispose();
        try
        {
            if (!process.HasExited)
                process.Kill();
        }
        catch (InvalidOperationException) { }
        catch (System.ComponentModel.Win32Exception) { }
        finally
        {
            process.Dispose();
        }
    }

    internal static string QuoteArgument(string value)
    {
        var quoted = new StringBuilder("\"");
        int slashes = 0;
        foreach (char character in value)
        {
            if (character == '\\')
            {
                slashes++;
                continue;
            }
            quoted.Append('\\', character == '"' ? slashes * 2 + 1 : slashes);
            quoted.Append(character);
            slashes = 0;
        }
        quoted.Append('\\', slashes * 2).Append('"');
        return quoted.ToString();
    }

    // Keep the engine tied to the desktop process even if the window terminates abruptly.
    private sealed class ChildJob : SafeHandleZeroOrMinusOneIsInvalid
    {
        private const uint KillOnClose = 0x00002000;
        private const int ExtendedLimitInformation = 9;

        private ChildJob() : base(true) { }

        public static ChildJob Create()
        {
            var handle = CreateJobObject(IntPtr.Zero, null);
            if (handle.IsInvalid)
            {
                handle.Dispose();
                return null;
            }
            var limits = new JobLimits();
            limits.Basic.LimitFlags = KillOnClose;
            if (SetInformationJobObject(handle, ExtendedLimitInformation,
                    ref limits, Marshal.SizeOf(typeof(JobLimits))))
                return handle;
            handle.Dispose();
            return null;
        }

        // Some managed environments prevent assigning a process to a nested job.
        // Normal window close still explicitly stops the owned process.
        public void Assign(Process child)
        {
            try { AssignProcessToJobObject(this, child.Handle); }
            catch (InvalidOperationException) { }
            catch (System.ComponentModel.Win32Exception) { }
        }

        protected override bool ReleaseHandle()
        {
            return CloseHandle(handle);
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct BasicLimits
        {
            public long ProcessUserTimeLimit, JobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize, MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass, SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct IoCounters
        {
            public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
            public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct JobLimits
        {
            public BasicLimits Basic;
            public IoCounters Io;
            public UIntPtr ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemory, PeakJobMemory;
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern ChildJob CreateJobObject(IntPtr attributes, string name);
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool SetInformationJobObject(
            ChildJob job, int informationClass, ref JobLimits information, int informationLength);
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool AssignProcessToJobObject(ChildJob job, IntPtr process);
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool CloseHandle(IntPtr handle);
    }
}
