; Per-user installer. No admin. Registers the Lucent hand-off and a Start menu shortcut.
#ifndef Source
  #define Source "..\..\dist\payload"
#endif
#ifndef AppVersion
  #define AppVersion "0.2.0-rc.6"
#endif

[Setup]
AppId={{A7E1C4B2-6D40-4F1A-9C3E-8B5D2F0E1A44}
AppName=Voxprint AI Enlarge
AppVersion={#AppVersion}
AppPublisher=Aleksandr Mitroshenkov
DefaultDirName={localappdata}\Voxprint\Enlarge
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\dist
OutputBaseFilename=VoxprintEnlarge-Setup
UninstallDisplayIcon={app}\assets\enlarge.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Files]
Source: "{#Source}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{userprograms}\Voxprint\Voxprint AI Enlarge"; Filename: "{app}\VoxprintEnlarge.exe"; WorkingDir: "{app}"; IconFilename: "{app}\assets\enlarge.ico"

; Post-install runs the bundled interpreter, not the launcher. The launcher
; prefers a shared Python that may be stale (no voxprint_common, no
; onnxruntime-gpu). A non-zero exit aborts Setup with a message.
[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  AppDir, Python, Main, Exe: String;
begin
  if CurStep <> ssPostInstall then
    Exit;
  AppDir := ExpandConstant('{app}');
  Python := AppDir + '\python\python.exe';
  Main := AppDir + '\main.py';
  Exe := AppDir + '\VoxprintEnlarge.exe';
  if not Exec(Python, '"' + Main + '" --install-suite', AppDir, SW_SHOWNORMAL, ewWaitUntilTerminated, ResultCode) then
  begin
    MsgBox('Voxprint AI Enlarge could not start the shared suite install.', mbError, MB_OK);
    Abort;
  end;
  if ResultCode <> 0 then
  begin
    MsgBox('Voxprint AI Enlarge could not install the shared suite (exit ' + IntToStr(ResultCode) + ').', mbError, MB_OK);
    Abort;
  end;
  if not Exec(Python, '"' + Main + '" --register --exe "' + Exe + '"', AppDir, SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    MsgBox('Voxprint AI Enlarge could not start registration.', mbError, MB_OK);
    Abort;
  end;
  if ResultCode <> 0 then
  begin
    MsgBox('Voxprint AI Enlarge could not register the app (exit ' + IntToStr(ResultCode) + ').', mbError, MB_OK);
    Abort;
  end;
end;

[UninstallRun]
Filename: "{app}\VoxprintEnlarge.exe"; Parameters: "--uninstall-suite"; Flags: waituntilterminated; RunOnceId: uninstallsuite
Filename: "{app}\VoxprintEnlarge.exe"; Parameters: "--unregister"; Flags: runhidden waituntilterminated; RunOnceId: unregister
