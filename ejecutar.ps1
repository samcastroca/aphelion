<#
.SYNOPSIS
    Prepara el entorno y construye la entrega completa, de principio a fin.

.DESCRIPTION
    Pensado para una máquina nueva: instala dependencias, detecta la GPU, pone
    la build de PyTorch que corresponda y lanza el pipeline.

    Antes de ejecutar hace falta una sola cosa manual: copiar el corpus de ADL a
        data\CORPUS CODEFEST AD ASTRA 2026\

    Sin parámetros hace lo correcto: los dos encoders, el backend que le
    corresponda al hardware y el lote que mejor rinda en él. Los parámetros están
    para desviarse de eso a propósito.

.PARAMETER Encoders
    Qué encoders indexar. Por defecto los dos. Con uno solo se llega antes a un
    índice utilizable, a costa de perder la fusión de los dos espacios.

.PARAMETER Backend
    torch (CUDA o CPU) u onnx (DirectML, GPU Radeon). Por defecto lo elige el
    hardware presente.

.PARAMETER Lote
    Fragmentos por lote de codificación. Por defecto 8 en DirectML y 128 en CUDA.
    Bájalo si la GPU se queda sin memoria.

.PARAMETER Reparto
    Porcentaje de los bloques que codifica esta máquina, como A:B. Se detiene
    ahí: no extrae, no fragmenta y no empaqueta, así que no necesita el corpus ni
    Tesseract — necesita el archivo trabajo\fragmentos.jsonl copiado de la
    máquina coordinadora, el mismo en las tres. Los tramos no tienen que ser
    iguales, quien tenga mejor GPU carga con más porcentaje, pero entre todos
    deben cubrir de 0 a 100 sin solaparse.

.EXAMPLE
    .\ejecutar.ps1
    .\ejecutar.ps1 -Desde 04_indexar:bge-m3    # reanudar tras un fallo
    .\ejecutar.ps1 -SoloEntorno                # preparar sin procesar nada

    .\ejecutar.ps1 -Encoders bge-m3            # un solo encoder
    .\ejecutar.ps1 -Encoders bge-m3,me5-large -Lote 64
    .\ejecutar.ps1 -Backend torch -Lote 32     # forzar el backend

    # Reparto entre tres máquinas, la primera con el doble de GPU que las otras:
    .\ejecutar.ps1 -Reparto 0:50               # en la máquina rápida
    .\ejecutar.ps1 -Reparto 50:75              # en la segunda
    .\ejecutar.ps1 -Reparto 75:100             # en la tercera
#>
[CmdletBinding()]
param(
    [string]$Desde,
    [string[]]$Encoders,
    [ValidateSet('auto', 'torch', 'onnx')]
    [string]$Backend,
    [ValidateRange(1, 1024)]
    [int]$Lote,
    [switch]$SoloEntorno,
    [switch]$SinOcr,
    [switch]$Forzar,
    [ValidatePattern('^\d+(\.\d+)?:\d+(\.\d+)?$')]
    [string]$Reparto
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

if ($Reparto) {
    # Quien solo codifica un tramo no parte del corpus, sino del archivo de
    # fragmentos que ya produjo la coordinadora. Regenerarlo aquí daría otro
    # archivo y sus vectores no encajarían con los de nadie.
    Paso "Fragmentos"
    $fragmentos = "trabajo\fragmentos.jsonl"
    if (-not (Test-Path $fragmentos)) {
        Write-Host "  Falta $fragmentos. Cópialo de la máquina coordinadora a:" -ForegroundColor Red
        Write-Host "    $PSScriptRoot\trabajo\" -ForegroundColor Red
        exit 1
    }
    Bien ("{0:N0} MB" -f ((Get-Item $fragmentos).Length / 1MB))
} else {
    Paso "Corpus"
    $corpus = "data\CORPUS CODEFEST AD ASTRA 2026"
    if (-not (Test-Path $corpus)) {
        Write-Host "  Falta el corpus de ADL. Cópialo a:" -ForegroundColor Red
        Write-Host "    $((Resolve-Path 'data').Path)\CORPUS CODEFEST AD ASTRA 2026\" -ForegroundColor Red
        exit 1
    }
    $n = (Get-ChildItem $corpus -Recurse -File | Measure-Object).Count
    Bien "$n archivos"
}

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

# Con reparto no se extrae nada, así que el OCR no entra en juego.
if (-not $Reparto) {
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
}

if ($SoloEntorno) {
    Paso "Entorno listo"
    Write-Host "  Lanza el proceso con: .\ejecutar.ps1" -ForegroundColor Green
    exit 0
}

Paso "Pipeline"
# Los nombres de encoder no se validan aquí: los define config.ENCODERS y es
# pipeline.py quien los comprueba, antes de tocar nada.
$argumentos = @("run", "python", "scripts/pipeline.py")
if ($Desde) { $argumentos += @("--desde", $Desde) }
if ($Encoders) { $argumentos += @("--encoders", ($Encoders -join ",")) }
if ($Backend) { $argumentos += @("--backend", $Backend) }
if ($Lote) { $argumentos += @("--lote", $Lote) }
if ($SinOcr) { $argumentos += "--sin-ocr" }
if ($Forzar) { $argumentos += "--forzar" }
if ($Reparto) { $argumentos += @("--reparto", $Reparto) }

& uv @argumentos
$codigo = $LASTEXITCODE

if ($Reparto -and $codigo -eq 0) {
    Paso "Tramo listo"
    Write-Host "  Manda a la máquina coordinadora los .npy de:" -ForegroundColor Green
    Get-ChildItem "trabajo\embeddings" -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Get-ChildItem $_.FullName -Directory } |
        ForEach-Object { Write-Host "    $($_.FullName)" -ForegroundColor Green }
    Write-Host "  Allí se juntan con los tramos de las otras y se arma el índice." -ForegroundColor Green
}

exit $codigo
