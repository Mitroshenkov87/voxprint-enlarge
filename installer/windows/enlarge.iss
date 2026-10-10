; Per-user installer. No admin. Registers the Lucent hand-off and a Start menu shortcut.
#ifndef Source
  #define Source "..\..\dist\payload"
#endif
#ifndef AppVersion
  #define AppVersion "0.2.0-rc.1"
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

[Run]
Filename: "{app}\VoxprintEnlarge.exe"; Parameters: "--register --exe ""{app}\VoxprintEnlarge.exe"""; Flags: runhidden waituntilterminated

[UninstallRun]
Filename: "{app}\VoxprintEnlarge.exe"; Parameters: "--unregister"; Flags: runhidden waituntilterminated; RunOnceId: unregister
