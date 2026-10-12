using System.Diagnostics;
using System.Text.Json;
#if GUI_LAUNCHER
using System.Runtime.InteropServices;
#endif
using System.Threading.Tasks;

string dir = AppContext.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
bool console = args.Length > 0;
#if GUI_LAUNCHER
// A redirected stdout or stderr is a pipe (PowerShell `& exe --version`).
// AttachConsole would send the child's prints to a console the caller does
// not capture, and $LASTEXITCODE stays empty. Inherited handles print and
// return the exit code. The console launcher is a console-subsystem exe, so
// it must not attach: that would open a second console.
if (console && !StdioRedirected())
{
    NativeConsole.AttachParent();
}
#endif
string python = ResolvePython(dir, console, args);
string main = Path.Combine(dir, "main.py");
if (!File.Exists(python) || !File.Exists(main))
{
    Console.Error.WriteLine("VOXPrint AI Enlarge is missing its Python runtime.");
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
    List<(string PointerDir, string Exe)> found = SharedInterpreters(console);
    if (found.Count == 0)
    {
        Log($"bundled python, shared runtime is absent: {bundled}");
        return bundled;
    }
    foreach ((string PointerDir, string Exe) candidate in found)
    {
        if (ImportsReady(candidate.PointerDir, candidate.Exe))
        {
            Log($"shared python, import check passed: {candidate.Exe}");
            return candidate.Exe;
        }
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

static List<(string PointerDir, string Exe)> SharedInterpreters(bool console)
{
    string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
    string shared = Path.Combine(local, "Voxprint", "shared");
    var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
    var found = new List<(string PointerDir, string Exe)>();
    foreach (string relative in ManifestPythonPaths(shared))
    {
        AddCandidates(found, seen, Path.Combine(shared, relative), console);
    }
    if (!Directory.Exists(shared))
    {
        return found;
    }
    foreach (string dir in Directory.GetDirectories(shared))
    {
        string name = Path.GetFileName(dir);
        bool pointer = name.Equals("python", StringComparison.OrdinalIgnoreCase)
            || name.StartsWith("python-", StringComparison.OrdinalIgnoreCase);
        if (!pointer)
        {
            continue;
        }
        AddCandidates(found, seen, dir, console);
    }
    return found;
}

static List<string> ManifestPythonPaths(string shared)
{
    var paths = new List<string>();
    string manifest = Path.Combine(shared, "manifest.json");
    if (!File.Exists(manifest))
    {
        return paths;
    }
    try
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifest));
        if (!document.RootElement.TryGetProperty("components", out JsonElement components)
            || components.ValueKind != JsonValueKind.Object)
        {
            return paths;
        }
        foreach (JsonProperty item in components.EnumerateObject())
        {
            JsonElement body = item.Value;
            if (body.ValueKind != JsonValueKind.Object)
            {
                continue;
            }
            string kind = body.TryGetProperty("kind", out JsonElement kindValue) ? kindValue.GetString() ?? "" : "";
            bool python = item.Name.StartsWith("python", StringComparison.OrdinalIgnoreCase) || kind == "runtime";
            if (!python || !body.TryGetProperty("path", out JsonElement pathValue))
            {
                continue;
            }
            string relative = pathValue.GetString() ?? "";
            if (relative.Length == 0 || Path.IsPathRooted(relative) || relative.Contains("..", StringComparison.Ordinal))
            {
                continue;
            }
            paths.Add(relative);
        }
    }
    catch (Exception ex) when (ex is JsonException or InvalidOperationException or IOException)
    {
        Log($"shared manifest could not be read: {ex.Message}");
    }
    return paths;
}

static void AddCandidates(List<(string PointerDir, string Exe)> found, HashSet<string> seen, string pointerDir, bool console)
{
    string fromJson = InterpreterExecutable(pointerDir);
    if (fromJson.Length > 0)
    {
        Remember(found, seen, pointerDir, PreferWindowless(fromJson, console));
        return;
    }
    string preferred = Path.Combine(pointerDir, console ? "python.exe" : "pythonw.exe");
    if (File.Exists(preferred))
    {
        Remember(found, seen, pointerDir, preferred);
        return;
    }
    string fallback = Path.Combine(pointerDir, "python.exe");
    if (File.Exists(fallback))
    {
        Remember(found, seen, pointerDir, fallback);
    }
}

static string InterpreterExecutable(string pointerDir)
{
    string pointer = Path.Combine(pointerDir, "interpreter.json");
    if (!File.Exists(pointer))
    {
        return "";
    }
    try
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(pointer));
        JsonElement root = document.RootElement;
        if (!root.TryGetProperty("executable", out JsonElement executableValue))
        {
            return "";
        }
        string executable = executableValue.GetString() ?? "";
        string version = root.TryGetProperty("version", out JsonElement versionValue) ? versionValue.GetString() ?? "" : "";
        bool python314 = version == "3.14" || version.StartsWith("3.14.", StringComparison.Ordinal);
        if (!python314 || executable.Length == 0 || !File.Exists(executable))
        {
            return "";
        }
        return executable;
    }
    catch (Exception ex) when (ex is JsonException or InvalidOperationException or IOException)
    {
        return "";
    }
}

static string PreferWindowless(string executable, bool console)
{
    if (console)
    {
        return executable;
    }
    string? dir = Path.GetDirectoryName(executable);
    if (dir is null)
    {
        return executable;
    }
    string windowless = Path.Combine(dir, "pythonw.exe");
    return File.Exists(windowless) ? windowless : executable;
}

static void Remember(List<(string PointerDir, string Exe)> found, HashSet<string> seen, string pointerDir, string executable)
{
    if (!seen.Add(executable))
    {
        return;
    }
    found.Add((pointerDir, executable));
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
    string ticks = ProbeTicks(pythonExe);
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

static string ProbeTicks(string pythonExe)
{
    // python.exe alone does not change when pip adds or removes a wheel.
    // The stamp follows the resolved interpreter, not the pointer folder.
    long ticks = File.GetLastWriteTimeUtc(pythonExe).Ticks;
    string site = SitePackages(pythonExe);
    if (Directory.Exists(site))
    {
        ticks ^= Directory.GetLastWriteTimeUtc(site).Ticks;
    }
    return ticks.ToString();
}

static string SitePackages(string pythonExe)
{
    string? dir = Path.GetDirectoryName(pythonExe);
    if (dir is null)
    {
        return "";
    }
    if (Path.GetFileName(dir).Equals("Scripts", StringComparison.OrdinalIgnoreCase)
        || Path.GetFileName(dir).Equals("bin", StringComparison.OrdinalIgnoreCase))
    {
        string? parent = Path.GetDirectoryName(dir);
        if (parent is not null)
        {
            return Path.Combine(parent, "Lib", "site-packages");
        }
    }
    return Path.Combine(dir, "Lib", "site-packages");
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

#if GUI_LAUNCHER
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
#endif
