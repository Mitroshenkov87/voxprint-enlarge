; Admin installer. The suite folder is under Program Files.
; An older per-user copy is removed without running its uninstaller.
#ifndef Source
  #define Source "..\..\dist\payload"
#endif
#ifndef AppVersion
  #define AppVersion "0.2.0-rc.6"
#endif

[Setup]
AppId={{A7E1C4B2-6D40-4F1A-9C3E-8B5D2F0E1A44}
AppName=VOXPrint AI Enlarge
AppVersion={#AppVersion}
AppPublisher=Aleksandr Mitroshenkov
DefaultDirName={autopf}\VOXPrint AI Suite\Enlarge
UsePreviousAppDir=no
DisableProgramGroupPage=yes
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\dist
OutputBaseFilename=voxprint-enlarge-setup
UninstallDisplayIcon={app}\assets\enlarge.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Files]
Source: "{#Source}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\VOXPrint AI Suite\VOXPrint AI Enlarge"; Filename: "{app}\voxprint-enlarge.exe"; WorkingDir: "{app}"; IconFilename: "{app}\assets\enlarge.ico"

; Post-install runs the bundled interpreter, not the launcher. The launcher
; prefers a shared Python that may be stale (no voxprint_common, no
; onnxruntime-gpu). --install-suite then calls VOXPrint MDL with the shared
; interpreter. A non-zero exit is written to enlarge-install.log and aborts
; Setup without a message box when /VERYSILENT or /SUPPRESSMSGBOXES is set.
; --register records the console exe so Canvas gets a native console process.
[Code]
const
  OldUninstallKey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{A7E1C4B2-6D40-4F1A-9C3E-8B5D2F0E1A44}_is1';

// Drop the per-user uninstall registration before Setup treats it as the
// previous install. Running that uninstaller would call --uninstall-suite.
function InitializeSetup(): Boolean;
begin
  RegDeleteKeyIncludingSubkeys(HKCU, OldUninstallKey);
  Result := True;
end;

// The old per-user uninstaller runs --uninstall-suite and would drop the
// shared runtime. Remove only the app folder, the old shortcut, and the
// old uninstall key. Models, shared Python, and logs stay under LocalAppData\Voxprint.
procedure MigratePerUserInstall();
var
  OldDir, Shortcut, Folder: String;
begin
  OldDir := ExpandConstant('{localappdata}\Voxprint\Enlarge');
  if DirExists(OldDir) then
    DelTree(OldDir, True, True, True);
  Shortcut := ExpandConstant('{userprograms}\Voxprint\Voxprint AI Enlarge.lnk');
  if FileExists(Shortcut) then
    DeleteFile(Shortcut);
  Folder := ExpandConstant('{userprograms}\Voxprint');
  if DirExists(Folder) then
    RemoveDir(Folder);
  RegDeleteKeyIncludingSubkeys(HKCU, OldUninstallKey);
end;

function InstallLog(): String;
begin
  Result := ExpandConstant('{localappdata}\Voxprint\logs\enlarge-install.log');
end;

procedure FailInstall(const Text: String);
begin
  Log(Text);
  if WizardSilent then
  begin
    Abort;
    Exit;
  end;
  SuppressibleMsgBox(Text, mbError, MB_OK, IDOK);
  Abort;
end;

function RunLogged(const FileName, Params, WorkingDir: String; var ResultCode: Integer): Boolean;
var
  LogFile, Command: String;
begin
  LogFile := InstallLog();
  ForceDirectories(ExtractFilePath(LogFile));
  Command := '"' + FileName + '" ' + Params + ' 1>> "' + LogFile + '" 2>&1';
  Result := Exec(ExpandConstant('{cmd}'), '/d /c "' + Command + '"', WorkingDir, SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  AppDir, Python, Main, Exe: String;
begin
  if CurStep = ssInstall then
  begin
    MigratePerUserInstall();
    Exit;
  end;
  if CurStep <> ssPostInstall then
    Exit;
  AppDir := ExpandConstant('{app}');
  Python := AppDir + '\python\python.exe';
  Main := AppDir + '\main.py';
  Exe := AppDir + '\voxprint-enlarge-cli.exe';
  if not RunLogged(Python, '"' + Main + '" --install-suite', AppDir, ResultCode) then
  begin
    FailInstall('VOXPrint AI Enlarge could not start the shared suite install. See ' + InstallLog());
    Exit;
  end;
  if ResultCode <> 0 then
  begin
    FailInstall('VOXPrint AI Enlarge could not install the shared suite (exit ' + IntToStr(ResultCode) + '). See ' + InstallLog());
    Exit;
  end;
  if not RunLogged(Python, '"' + Main + '" --register --exe "' + Exe + '"', AppDir, ResultCode) then
  begin
    FailInstall('VOXPrint AI Enlarge could not start registration. See ' + InstallLog());
    Exit;
  end;
  if ResultCode <> 0 then
  begin
    FailInstall('VOXPrint AI Enlarge could not register the app (exit ' + IntToStr(ResultCode) + '). See ' + InstallLog());
  end;
end;

[UninstallRun]
Filename: "{app}\voxprint-enlarge-cli.exe"; Parameters: "--uninstall-suite"; Flags: waituntilterminated; RunOnceId: uninstallsuite
Filename: "{app}\voxprint-enlarge-cli.exe"; Parameters: "--unregister"; Flags: runhidden waituntilterminated; RunOnceId: unregister
