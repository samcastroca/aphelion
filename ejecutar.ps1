<#
.SYNOPSIS
    Prepara el entorno y construye la entrega completa, de principio a fin.

.DESCRIPTION
    Pensado para una máquina nueva: instala dependencias, detecta la GPU, pone
    la build de PyTorch que corresponda y lanza el pipeline.

    Antes de ejecutar hace falta una sola cosa manual: copiar el corpus de ADL a
        data\CORPUS CODEFEST AD ASTRA 2026\

.EXAMPLE
    .\ejecutar.ps1
    .\ejecutar.ps1 -Desde 04_indexar:bge-m3    # reanudar tras un fallo
    .\ejecutar.ps1 -SoloEntorno                # preparar sin procesar nada
#>
[CmdletBinding()]
param(
    [string]$Desde,
    [switch]$SoloEntorno,
    [switch]$SinOcr
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Paso($texto) { Write-Host "`n=== $texto ===" -ForegroundColor Cyan }
function Aviso($texto) { Write-Host "  ! $texto" -ForegroundColor Yellow }
function Bien($texto) { Write-Host "  $texto" -ForegroundColor Green }

Paso "Comprobando uv"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Aviso "uv no está instalado. Instalándolo..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv no quedó en el PATH. Abre una terminal nueva y repite."
    }
}
Bien (uv --version)

Paso "Corpus"
$corpus = "data\CORPUS CODEFEST AD ASTRA 2026"
if (-not (Test-Path $corpus)) {
    Write-Host "  Falta el corpus de ADL. Cópialo a:" -ForegroundColor Red
    Write-Host "    $((Resolve-Path 'data').Path)\CORPUS CODEFEST AD ASTRA 2026\" -ForegroundColor Red
    exit 1
}
$n = (Get-ChildItem $corpus -Recurse -File | Measure-Object).Count
Bien "$n archivos"

Paso "GPU y dependencias"
# El extra decide de qué índice sale torch, así que la GPU se detecta antes de
# sincronizar: instalar torch aparte no serviría, porque el `uv run` siguiente
# vuelve a alinear el entorno con el lock y lo reemplazaría.
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$radeon = if ($nvidia) { $null } else {
    Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "Radeon" }
}

if ($nvidia) {
    Bien "NVIDIA detectada: $(& nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1)"
    uv sync --extra cuda
} elseif ($radeon) {
    Aviso "Radeon detectada: $($radeon.Name). Usando ONNX Runtime + DirectML."
    uv sync --extra amd
} else {
    Aviso "Sin GPU detectada. La codificación irá por CPU y tardará días."
    uv sync
}
if ($LASTEXITCODE -ne 0) { throw "uv sync falló" }

if ($nvidia) {
    $cuda = uv run python -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ($cuda -eq "True") {
        Bien (uv run python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))")
    } else {
        Aviso "PyTorch no ve la GPU pese a la build CUDA. Revisa el driver."
    }
}

Paso "Tesseract (OCR)"
$cacheOcr = "data\ocr.jsonl"
$enCache = if (Test-Path $cacheOcr) { (Get-Content $cacheOcr | Measure-Object -Line).Lines } else { 0 }
if (Get-Command tesseract -ErrorAction SilentlyContinue) {
    Bien (tesseract --version | Select-Object -First 1)
} elseif (Test-Path "C:\Program Files\Tesseract-OCR\tesseract.exe") {
    Bien "instalado en Program Files"
} else {
    Aviso "Tesseract no está. $enCache documentos vienen ya reconocidos en la caché."
    Aviso "Para los que falten: winget install UB-Mannheim.TesseractOCR (con el paquete 'spa')"
}

if ($SoloEntorno) {
    Paso "Entorno listo"
    Write-Host "  Lanza el proceso con: .\ejecutar.ps1" -ForegroundColor Green
    exit 0
}

Paso "Pipeline"
$argumentos = @("run", "python", "scripts/pipeline.py")
if ($Desde) { $argumentos += @("--desde", $Desde) }
if ($SinOcr) { $argumentos += "--sin-ocr" }

& uv @argumentos
exit $LASTEXITCODE
