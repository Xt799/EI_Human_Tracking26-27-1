# Open X-AnyLabeling on one image folder of the dataset with this project's labels and digit shortcuts.
#
#   powershell -File training/labeling/label.ps1 dataset/detection/val/recordings
#
# X-AnyLabeling is not part of this repository. Set $env:XANYLABELING to its executable, or it is looked up in
# D:\lali\Workspace\X-Anylabeling\.venv-cu12\Scripts\xanylabeling.exe. It is started through xanylabeling_heading.py
# with the Python next to that executable, which adds a live heading arrow drawn from the front and rear points.
# Settings changed in the app are saved to .xanylabeling/ in the repository (git-ignored), not to your home folder.
param([Parameter(Mandatory = $true)][string]$Folder)

$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$exe = $env:XANYLABELING
if (-not $exe) { $exe = "D:\lali\Workspace\X-Anylabeling\.venv-cu12\Scripts\xanylabeling.exe" }
if (-not (Test-Path $exe)) { throw "X-AnyLabeling not found: $exe (set `$env:XANYLABELING)" }

$images = Resolve-Path (Join-Path $repo $Folder)
$workDir = Join-Path $repo ".xanylabeling"
New-Item -ItemType Directory -Force $workDir | Out-Null

$python = Join-Path (Split-Path $exe) "python.exe"
& $python (Join-Path $PSScriptRoot "xanylabeling_heading.py") --work-dir $workDir --config (Join-Path $PSScriptRoot "xanylabeling.yaml") --filename $images --output $images --nodata --autosave
