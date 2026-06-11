#define MyAppName "Reefscape RL"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Synapse contributors"
#define MyAppExeName "ReefscapeRL.exe"

[Setup]
AppId={{A7F06B52-39B5-4B4F-9B7C-3C489BA24396}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\Reefscape RL
DefaultGroupName=Reefscape RL
DisableProgramGroupPage=yes
OutputDir=..\builds
OutputBaseFilename=ReefscapeRL-Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "..\builds\ReefscapeRL\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\builds\prereqs\python-3.13.13-amd64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
Name: "{group}\Reefscape RL"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\Reefscape RL"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{group}\Setup Python Environment"; Filename: "{app}\Setup Python Environment.bat"; WorkingDir: "{app}"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\setup_venv.ps1"" -BootstrapPython -PythonInstaller ""{tmp}\python-3.13.13-amd64.exe"""; StatusMsg: "Installing Python environment..."; Flags: waituntilterminated
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Reefscape RL"; Flags: nowait postinstall skipifsilent; WorkingDir: "{app}"
