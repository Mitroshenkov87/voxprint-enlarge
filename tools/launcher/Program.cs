using System.Diagnostics;
using System.Runtime.InteropServices;

string dir = AppContext.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
bool console = args.Length > 0;
if (console)
{
    NativeConsole.AttachParent();
}
string python = ResolvePython(dir, console);
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

static string ResolvePython(string dir, bool console)
{
    string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
    string shared = Path.Combine(local, "Voxprint", "shared", "python");
    string sharedPreferred = Path.Combine(shared, console ? "python.exe" : "pythonw.exe");
    if (File.Exists(sharedPreferred))
    {
        return sharedPreferred;
    }
    string sharedFallback = Path.Combine(shared, "python.exe");
    if (File.Exists(sharedFallback))
    {
        return sharedFallback;
    }
    string folder = Path.Combine(dir, "python");
    string bundled = Path.Combine(folder, console ? "python.exe" : "pythonw.exe");
    if (!File.Exists(bundled))
    {
        bundled = Path.Combine(folder, "python.exe");
    }
    return bundled;
}

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
