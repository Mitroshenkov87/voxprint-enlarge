using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

string dir = AppContext.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
bool console = args.Length > 0;
// A redirected stdout or stderr is a pipe (PowerShell `& exe --version`).
// AttachConsole would send the child's prints to a console the caller does
// not capture, and $LASTEXITCODE stays empty. Inherited handles print and
// return the exit code.
if (console && !StdioRedirected())
{
    NativeConsole.AttachParent();
}
string python = ResolvePython(dir, console, args);
string main = Path.Combine(dir, "main.py");
if (!File.Exists(python) || !File.Exists(main))
{
    Console.Error.WriteLine("Voxprint AI Enlarge is missing its Python runtime.");
    return 1;
}
var start = new ProcessStartInfo(python)
{
    UseShellExecute = false,
    WorkingDirectory = dir,
    CreateNoWindow = !console,
};
start.ArgumentList.Add(main);
foreach (string arg in args)
{
    start.ArgumentList.Add(arg);
}
using Process process = Process.Start(start) ?? throw new InvalidOperationException("python did not start");
process.WaitForExit();
return process.ExitCode;

static string ResolvePython(string dir, bool console, string[] args)
{
    string bundled = BundledPython(dir, console);
    if (UsesBundledInterpreter(args))
    {
        Log($"bundled python for {string.Join(' ', args)}: {bundled}");
        return bundled;
    }
    string sharedDir = SharedPythonDir();
    string shared = SharedPython(sharedDir, console);
    if (shared.Length == 0)
    {
        Log($"bundled python, shared runtime is absent: {bundled}");
        return bundled;
    }
    if (ImportsReady(sharedDir, Path.Combine(sharedDir, "python.exe")))
    {
        Log($"shared python, import check passed: {shared}");
        return shared;
    }
    Log($"bundled python, shared import check failed: {bundled}");
    return bundled;
}

static bool UsesBundledInterpreter(string[] args)
{
    foreach (string arg in args)
    {
        if (arg is "--install-suite" or "--register" or "--uninstall-suite" or "--unregister")
        {
            return true;
        }
    }
    return false;
}

static string SharedPythonDir()
{
    string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
    return Path.Combine(local, "Voxprint", "shared", "python");
}

static string SharedPython(string shared, bool console)
{
    string preferred = Path.Combine(shared, console ? "python.exe" : "pythonw.exe");
    if (File.Exists(preferred))
    {
        return preferred;
    }
    string fallback = Path.Combine(shared, "python.exe");
    return File.Exists(fallback) ? fallback : "";
}

static string BundledPython(string dir, bool console)
{
    string folder = Path.Combine(dir, "python");
    string bundled = Path.Combine(folder, console ? "python.exe" : "pythonw.exe");
    if (!File.Exists(bundled))
    {
        bundled = Path.Combine(folder, "python.exe");
    }
    return bundled;
}

static bool ImportsReady(string sharedDir, string pythonExe)
{
    if (!File.Exists(pythonExe))
    {
        return false;
    }
    string stampPath = Path.Combine(sharedDir, ".enlarge-import-probe");
    string ticks = ProbeTicks(sharedDir, pythonExe);
    if (File.Exists(stampPath))
    {
        string text = File.ReadAllText(stampPath).Trim();
        if (text == ticks)
        {
            Log("shared import check cached");
            return true;
        }
    }
    var start = new ProcessStartInfo(pythonExe)
    {
        UseShellExecute = false,
        CreateNoWindow = true,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
    };
    start.ArgumentList.Add("-c");
    start.ArgumentList.Add("import voxprint_common, voxprint_engine, onnxruntime");
    try
    {
        using Process process = Process.Start(start) ?? throw new InvalidOperationException("python did not start");
        // Read on background tasks so a hung import can still hit the timeout.
        Task<string> outputTask = process.StandardOutput.ReadToEndAsync();
        Task<string> errorTask = process.StandardError.ReadToEndAsync();
        if (!process.WaitForExit(20000))
        {
            process.Kill(entireProcessTree: true);
            Log("shared import check timed out");
            return false;
        }
        if (!Task.WaitAll(new Task[] { outputTask, errorTask }, 5000))
        {
            Log("shared import check timed out");
            return false;
        }
        if (process.ExitCode == 0)
        {
            Directory.CreateDirectory(sharedDir);
            File.WriteAllText(stampPath, ticks);
            return true;
        }
        string error = errorTask.Result;
        string output = outputTask.Result;
        string reason = (error.Length > 0 ? error : output).Trim();
        if (reason.Length > 400)
        {
            reason = reason[..400];
        }
        Log($"shared import check exit {process.ExitCode}: {reason}");
        return false;
    }
    catch (Exception ex) when (ex is InvalidOperationException or System.ComponentModel.Win32Exception or IOException)
    {
        Log($"shared import check failed: {ex.Message}");
        return false;
    }
}

static string ProbeTicks(string sharedDir, string pythonExe)
{
    // python.exe alone does not change when pip adds or removes a wheel.
    long ticks = File.GetLastWriteTimeUtc(pythonExe).Ticks;
    string site = Path.Combine(sharedDir, "Lib", "site-packages");
    if (Directory.Exists(site))
    {
        ticks ^= Directory.GetLastWriteTimeUtc(site).Ticks;
    }
    return ticks.ToString();
}

static void Log(string message)
{
    string line = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + " " + message;
    try
    {
        Console.Error.WriteLine(line);
    }
    catch (IOException)
    {
        // The GUI has no console. The file below is the record.
    }
    try
    {
        string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        string folder = Path.Combine(local, "Voxprint", "logs");
        Directory.CreateDirectory(folder);
        File.AppendAllText(Path.Combine(folder, "enlarge-launcher.log"), line + Environment.NewLine);
    }
    catch (IOException)
    {
        // Logging must not stop the launch.
    }
}

static bool StdioRedirected()
{
    return !HandleIsConsole(-11) || !HandleIsConsole(-12);
}

static bool HandleIsConsole(int standard)
{
    IntPtr handle = GetStdHandle(standard);
    if (handle == IntPtr.Zero || handle == new IntPtr(-1))
    {
        return false;
    }
    return GetConsoleMode(handle, out _);
}

[DllImport("kernel32.dll", SetLastError = true)]
static extern IntPtr GetStdHandle(int nStdHandle);

[DllImport("kernel32.dll", SetLastError = true)]
static extern bool GetConsoleMode(IntPtr handle, out uint mode);

static class NativeConsole
{
    public static void AttachParent()
    {
        const int parent = -1;
        if (!AttachConsole(parent))
        {
            AllocConsole();
        }
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool AttachConsole(int processId);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool AllocConsole();
}
