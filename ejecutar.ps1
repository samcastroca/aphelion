<#
.SYNOPSIS
    Prepara el entorno y construye la entrega completa, de principio a fin.

.DESCRIPTION
    Pensado para una máquina nueva: instala dependencias, detecta la GPU, pone
    la build de PyTorch que corresponda y lanza el pipeline.

    Antes de ejecutar hace falta una sola cosa manual: copiar el corpus de ADL a
        datos\CORPUS CODEFEST AD ASTRA 2026\

.EXAMPLE
    .\ejecutar.ps1
    .\ejecutar.ps1 -Desde 03_indexar:bge-m3    # reanudar tras un fallo
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
$corpus = "datos\CORPUS CODEFEST AD ASTRA 2026"
if (-not (Test-Path $corpus)) {
    Write-Host "  Falta el corpus de ADL. Cópialo a:" -ForegroundColor Red
    Write-Host "    $((Resolve-Path 'datos').Path)\CORPUS CODEFEST AD ASTRA 2026\" -ForegroundColor Red
    exit 1
}
$n = (Get-ChildItem $corpus -Recurse -File | Measure-Object).Count
Bien "$n archivos"

Paso "Dependencias"
uv sync
if ($LASTEXITCODE -ne 0) { throw "uv sync falló" }

Paso "GPU"
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
    $gpu = (& nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1)
    Bien "NVIDIA detectada: $gpu"

    $cuda = uv run python -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ($cuda -ne "True") {
        Aviso "PyTorch no ve la GPU. Instalando la build CUDA 12.8..."
        # cu128 o superior es obligatorio en Blackwell (sm_120); las anteriores
        # fallan con un 'no kernel image is available' que despista.
        uv pip install torch --index-url https://download.pytorch.org/whl/cu128
        $cuda = uv run python -c "import torch; print(torch.cuda.is_available())" 2>$null
    }
    if ($cuda -eq "True") {
        Bien (uv run python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))")
    } else {
        Aviso "PyTorch sigue sin ver la GPU. Revisa el driver o la versión de Python."
    }
} else {
    $radeon = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "Radeon" }
    if ($radeon) {
        Aviso "Radeon detectada: $($radeon.Name). Usando ONNX Runtime + DirectML."
        uv sync --extra amd
    } else {
        Aviso "Sin GPU detectada. La codificación irá por CPU y tardará días."
    }
}

Paso "Tesseract (OCR)"
$cacheOcr = "datos\ocr.jsonl"
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
