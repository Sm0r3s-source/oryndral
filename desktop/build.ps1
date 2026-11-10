param(
    [string]$OutDir = (Join-Path $PSScriptRoot 'bin'),
    [string]$WebView2Directory,
    [string]$Compiler,
    [string]$WebView2Loader
)

$ErrorActionPreference = 'Stop'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = if (Test-Path -LiteralPath $vswhere) {
    & $vswhere -latest -products '*' -property installationPath | Select-Object -First 1
}
if (!$Compiler -and $vs) {
    $Compiler = Join-Path $vs 'MSBuild\Current\Bin\Roslyn\csc.exe'
}
if (!$WebView2Directory -and $vs) {
    $WebView2Directory = Join-Path $vs 'Common7\IDE\PrivateAssemblies'
}
if (!$Compiler -or !(Test-Path -LiteralPath $Compiler -PathType Leaf)) {
    throw 'Provide -Compiler with the path to Roslyn csc.exe from Visual Studio Build Tools.'
}
if (!$WebView2Directory -or !(Test-Path -LiteralPath $WebView2Directory -PathType Container)) {
    throw 'Provide -WebView2Directory containing the WebView2 SDK Core and Wpf assemblies and x64 loader.'
}

$framework = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319'
$libraries = @('System.dll', 'System.Core.dll', 'System.Xaml.dll', 'System.Net.Http.dll')
$references = @($libraries | ForEach-Object { Join-Path $framework $_ })
$references += @('WindowsBase.dll', 'PresentationCore.dll', 'PresentationFramework.dll' |
    ForEach-Object { Join-Path $framework "WPF\$_" })
$managed = @('Microsoft.Web.WebView2.Core.dll', 'Microsoft.Web.WebView2.Wpf.dll')
$references += @($managed | ForEach-Object { Join-Path $WebView2Directory $_ })
$missing = @($references | Where-Object { !(Test-Path -LiteralPath $_ -PathType Leaf) })
if ($missing.Count -gt 0) {
    throw "Missing build dependencies: $($missing -join ', ')"
}

function Test-X64Loader([string]$Path) {
    if (!$Path -or !(Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    $stream = [IO.File]::OpenRead($Path)
    $reader = New-Object IO.BinaryReader($stream)
    try {
        if ($stream.Length -lt 64 -or $reader.ReadUInt16() -ne 0x5A4D) { return $false }
        $stream.Position = 60
        $header = $reader.ReadInt32()
        if ($header -lt 0 -or $header -gt $stream.Length - 6) { return $false }
        $stream.Position = $header
        return ($reader.ReadUInt32() -eq 0x4550 -and $reader.ReadUInt16() -eq 0x8664)
    }
    finally {
        $reader.Dispose()
    }
}
$loaderCandidates = @($WebView2Loader,
    (Join-Path $WebView2Directory 'WebView2Loader.dll'),
    (Join-Path $WebView2Directory 'runtimes\win-x64\native\WebView2Loader.dll'))
if ($vs) {
    $loaderCandidates += Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\Markdown\runtimes\win-x64\native\WebView2Loader.dll'
}
$sdkVersion = (Get-Item -LiteralPath (Join-Path $WebView2Directory $managed[0])).VersionInfo.FileVersion
$loader = $loaderCandidates | Where-Object {
    (Test-X64Loader $_) -and (Get-Item -LiteralPath $_).VersionInfo.FileVersion -eq $sdkVersion
} | Select-Object -First 1
if (!$loader) {
    throw "Provide -WebView2Loader with the x64 WebView2Loader.dll matching SDK $sdkVersion."
}

$null = New-Item -ItemType Directory -Force -Path $OutDir
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path
$exe = Join-Path $OutDir 'Oryndral Desktop.exe'
$icon = Join-Path $PSScriptRoot 'Oryndral.ico'
$arguments = @('/nologo', '/target:winexe', '/platform:x64', '/langversion:7.3', '/optimize+', '/warnaserror+',
    "/win32manifest:$PSScriptRoot\app.manifest", "/win32icon:$icon", "/resource:$icon,Oryndral.ico", "/out:$exe")
$arguments += @($references | ForEach-Object { "/reference:$_" })
$arguments += @('Desktop.cs', 'ManagedServer.cs', 'LocalFiles.cs' |
    ForEach-Object { Join-Path $PSScriptRoot $_ })
& $Compiler @arguments
if ($LASTEXITCODE -ne 0) { throw "Desktop compilation failed with exit code $LASTEXITCODE." }

foreach ($name in $managed) {
    Copy-Item -LiteralPath (Join-Path $WebView2Directory $name) -Destination $OutDir -Force
}
Copy-Item -LiteralPath $loader -Destination (Join-Path $OutDir 'WebView2Loader.dll') -Force
Get-Item -LiteralPath $exe
