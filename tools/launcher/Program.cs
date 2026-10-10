using System.Diagnostics;

string dir = AppContext.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
bool console = args.Length > 0;
string folder = Path.Combine(dir, "python");
string python = Path.Combine(folder, console ? "python.exe" : "pythonw.exe");
if (!File.Exists(python))
{
    python = Path.Combine(folder, "python.exe");
}
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
};
start.ArgumentList.Add(main);
foreach (string arg in args)
{
    start.ArgumentList.Add(arg);
}
using Process process = Process.Start(start) ?? throw new InvalidOperationException("python did not start");
process.WaitForExit();
return process.ExitCode;
