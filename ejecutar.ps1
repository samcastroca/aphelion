<#
.SYNOPSIS
    Prepara el entorno y construye la entrega completa, de principio a fin.

.DESCRIPTION
    Pensado para una máquina nueva: instala dependencias, detecta la GPU, pone
    la build de PyTorch que corresponda y lanza el pipeline.

    Antes de ejecutar hace falta una sola cosa manual: copiar el corpus de ADL a
        data\CORPUS CODEFEST AD ASTRA 2026\

    Sin parámetros pregunta qué hacer con un menú, para no tener que recordar
    ninguna opción. Con cualquier parámetro —o con -Auto— no pregunta nada y
    corre directo, que es lo que necesitan la documentación y una tarea
    programada. Sin consola tampoco pregunta: toma los valores por defecto, que
    son los dos encoders, el backend que le corresponda al hardware y el lote que
    mejor rinda en él.

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
    Porcentaje de los bloques que codifica esta máquina, como A:B. Prepara los
    fragmentos si no están, codifica su tramo y se detiene: no empaqueta, porque
    el índice no está completo hasta juntar los tramos de todas.

    Los tramos no tienen que ser iguales —quien tenga mejor GPU carga con más
    porcentaje— pero entre todos deben cubrir de 0 a 100 sin solaparse.

    Cada máquina genera su propio trabajo\fragmentos.jsonl y sale idéntico, así
    que no hay que mover 285 MB. Compruébalo antes de codificar:
        uv run python scripts/etapas/04_indexar.py --huella

.EXAMPLE
    .\ejecutar.ps1                             # menú: elige y listo
    .\ejecutar.ps1 -Auto                       # sin preguntar, todo por defecto
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
    [string]$Reparto,
    [switch]$Auto
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Paso($texto) { Write-Host "`n=== $texto ===" -ForegroundColor Cyan }
function Aviso($texto) { Write-Host "  ! $texto" -ForegroundColor Yellow }
function Bien($texto) { Write-Host "  $texto" -ForegroundColor Green }

function Leer($pregunta, $porDefecto) {
    # Sin consola —una tarea programada, una tubería— Read-Host no lee nada. Que
    # devuelva el valor por defecto y no reviente: sin parámetros el guion tiene
    # que seguir construyendo la entrega entera, como siempre.
    try { $r = Read-Host "  $pregunta [$porDefecto]" } catch { return $porDefecto }
    if ([string]::IsNullOrWhiteSpace($r)) { return $porDefecto }
    return $r.Trim()
}

function Elegir($titulo, [string[]]$opciones, $porDefecto = 1) {
    Paso $titulo
    for ($i = 0; $i -lt $opciones.Count; $i++) {
        Write-Host ("  {0}) {1}" -f ($i + 1), $opciones[$i])
    }
    while ($true) {
        $r = Leer "Elige" $porDefecto
        if ($r -match '^\d+$' -and [int]$r -ge 1 -and [int]$r -le $opciones.Count) {
            return [int]$r
        }
        Aviso "Escribe un número entre 1 y $($opciones.Count)."
    }
}

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

Paso "GPU"
# Se detecta antes que nada porque decide dos cosas: de qué índice sale torch, y
# qué le muestra el menú a quien esté delante.
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$radeon = if ($nvidia) { $null } else {
    Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "Radeon" }
}

if ($nvidia) {
    $gpu = (& nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1)
    Bien "$gpu  ->  CUDA"
} elseif ($radeon) {
    Bien "$($radeon.Name)  ->  ONNX Runtime + DirectML"
} else {
    Aviso "Sin GPU. La codificación irá por CPU y tardará días."
}

# El menú aparece solo cuando nadie pidió nada por parámetro. Con cualquier
# parámetro —o con -Auto— el guion corre sin preguntar, que es lo que necesitan
# una tarea programada y la documentación.
if ($PSBoundParameters.Count -eq 0 -and -not $Auto) {
    switch (Elegir "Qué quieres hacer" @(
            "Construir la entrega completa                    (lo normal)"
            "Codificar solo mi parte, repartiendo entre varias PCs"
            "Reanudar desde una etapa                         (tras un fallo)"
            "Solo preparar el entorno, sin procesar nada"
        )) {
        2 {
            $maquinas = 1 + (Elegir "Entre cuántas máquinas se reparte" @(
                    "2 máquinas"
                    "3 máquinas"
                    "4 máquinas"
                    "5 máquinas"
                ) 2)

            # Las máquinas no rinden igual, así que los tramos tampoco tienen por
            # qué serlo. Cada perfil son pesos relativos: 2,1,1 es "la primera
            # rinde el doble que las otras dos". Los porcentajes salen de ahí, y
            # como el reparto se redondea sobre el número de bloques, tramos
            # contiguos cubren todo exactamente una vez — sean iguales o no.
            $perfiles = @(
                @{ nombre = "Todas parecidas";                pesos = @(1) * $maquinas }
                @{ nombre = "Una el doble que las demás";     pesos = @(2) + @(1) * ($maquinas - 1) }
                @{ nombre = "Una el triple que las demás";    pesos = @(3) + @(1) * ($maquinas - 1) }
                @{ nombre = "Escalonadas, de más a menos";    pesos = $maquinas..1 }
            )

            $opcionesPerfil = @()
            foreach ($p in $perfiles) {
                # Ojo con el nombre: PowerShell no distingue mayúsculas, así que
                # una variable `$reparto` sería el parámetro `$Reparto` y su
                # ValidatePattern rechazaría este texto.
                $suma = ($p.pesos | Measure-Object -Sum).Sum
                $porcentajes = ($p.pesos | ForEach-Object { "{0:0}%" -f ($_ * 100 / $suma) }) -join " / "
                $opcionesPerfil += ("{0,-28}  {1}" -f $p.nombre, $porcentajes)
            }
            $opcionesPerfil += "Otro reparto, a mano"

            $perfil = Elegir "Cómo se reparte la carga" $opcionesPerfil 1

            if ($perfil -le $perfiles.Count) {
                $pesos = $perfiles[$perfil - 1].pesos
                $suma = ($pesos | Measure-Object -Sum).Sum
                $tramos = @()
                $acumulado = 0
                foreach ($peso in $pesos) {
                    $ini = [math]::Round($acumulado * 100 / $suma, 3)
                    $acumulado += $peso
                    $fin = [math]::Round($acumulado * 100 / $suma, 3)
                    $tramos += "${ini}:${fin}"
                }
            } else {
                $tramos = @()
            }

            $etiquetas = @()
            for ($i = 0; $i -lt $tramos.Count; $i++) {
                $trozo = $tramos[$i] -split ":"
                $tamano = [double]$trozo[1] - [double]$trozo[0]
                $etiquetas += ("{0,-18}  la {1}ª,  {2:0}% del corpus" -f $tramos[$i], ($i + 1), $tamano)
            }
            $etiquetas += "escribir el tramo a mano"

            $elegido = Elegir "Qué tramo codifica esta máquina" $etiquetas 1
            if ($elegido -le $tramos.Count) {
                $Reparto = $tramos[$elegido - 1]
            } else {
                # A mano para el reparto que no encaje en ningún perfil. Entre
                # todas las máquinas los tramos deben cubrir de 0 a 100 sin
                # solaparse; cada una escribe el suyo.
                #
                # Se valida aquí y no más adelante: escrito a mano, el tramo se
                # salta el ValidatePattern del parámetro, y un `0-70` en vez de
                # `0:70` no se vería hasta que el indexador hubiera cargado el
                # modelo.
                # Se lee en una variable aparte y no en $Reparto: el parámetro
                # lleva un ValidatePattern, y asignarle un intento fallido
                # revienta el guion antes de poder rechazarlo con un mensaje.
                $sugerido = if ($tramos.Count) { $tramos[0] } else { "0:50" }
                while ($true) {
                    $tramoManual = Leer "Tramo A:B (0 a 100)" $sugerido
                    if ($tramoManual -match '^(\d+(\.\d+)?):(\d+(\.\d+)?)$') {
                        $a = [double]$Matches[1]
                        $b = [double]$Matches[3]
                        if ($a -lt $b -and $b -le 100) { break }
                        Aviso "El primero tiene que ser menor que el segundo, y el segundo como mucho 100."
                    } else {
                        Aviso "Formato A:B, por ejemplo 0:70."
                    }
                }
                $Reparto = $tramoManual
            }
            Bien "Tramo $Reparto"
        }
        3 {
            # Los nombres los define pipeline.py; si alguno dejara de existir, es
            # él quien lo dice y lista los válidos antes de tocar nada.
            $etapas = @(
                "01_extraer", "02_ocr", "03_fragmentar",
                "04_indexar", "05_empaquetar", "06_verificar"
            )
            $elegida = Elegir "Desde qué etapa" @(
                "Extracción del texto"
                "OCR de los escaneados"
                "Limpieza y fragmentación"
                "Codificación e índice   (la etapa larga)"
                "Empaquetado de la entrega"
                "Verificación del entregable"
            ) 4
            $Desde = $etapas[$elegida - 1]
        }
        4 { $SoloEntorno = [switch]$true }
    }

    if (-not $SoloEntorno) {
        $Encoders = switch (Elegir "Qué encoders indexar" @(
                "Los dos: bge-m3 y me5-large   (mejor recuperación, el doble de tiempo)"
                "Solo bge-m3                   (la mitad de tiempo)"
                "Solo me5-large"
            )) {
            2 { @("bge-m3") }
            3 { @("me5-large") }
            default { $null }
        }
        # Al reanudar desde la codificación hay que decirle a qué encoder, o
        # `--desde 04_indexar` no encaja con ninguna etapa del plan.
        if ($Desde -eq "04_indexar") {
            $primero = if ($Encoders) { $Encoders[0] } else { "bge-m3" }
            $Desde = "04_indexar:$primero"
        }
    }
}

if ($Reparto) {
    # Quien solo codifica un tramo no parte del corpus, sino del archivo de
    # fragmentos que ya produjo la coordinadora. Regenerarlo aquí daría otro
    # archivo y sus vectores no encajarían con los de nadie.
    Paso "Fragmentos"
    $fragmentos = "trabajo\fragmentos.jsonl"
    if (Test-Path $fragmentos) {
        Bien ("{0:N0} MB" -f ((Get-Item $fragmentos).Length / 1MB))
    } elseif (Test-Path "data\CORPUS CODEFEST AD ASTRA 2026") {
        # Todas las máquinas tienen el corpus, así que cada una genera el suyo y
        # no hay que mover 285 MB. Sale idéntico porque la fragmentación es
        # determinista; lo que las separaría es que una se saltara el OCR, y la
        # huella que se imprime luego lo delata antes de codificar nada.
        Aviso "No están hechos todavía. Se preparan aquí antes de codificar el tramo."
    } else {
        Write-Host "  Falta $fragmentos y no está el corpus para generarlo." -ForegroundColor Red
        Write-Host "  Copia el corpus a data\, o trae $fragmentos de otra máquina." -ForegroundColor Yellow
        exit 1
    }
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

Paso "Dependencias"
# El extra decide de qué índice sale torch. Instalarlo aparte no serviría: el
# `uv run` siguiente vuelve a alinear el entorno con el lock y lo reemplazaría.
if ($nvidia) {
    uv sync --extra cuda
} elseif ($radeon) {
    uv sync --extra amd
} else {
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
    Write-Host "  Comprueba que la huella coincide con la de las otras máquinas:" -ForegroundColor Green
    Write-Host "    uv run python scripts/etapas/04_indexar.py --huella" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Manda a la máquina que arma el índice los .npy de:" -ForegroundColor Green
    Get-ChildItem "trabajo\embeddings" -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Get-ChildItem $_.FullName -Directory } |
        ForEach-Object { Write-Host "    $($_.FullName)" -ForegroundColor Green }
    Write-Host "  Allí se juntan con los tramos de las otras y se arma el índice." -ForegroundColor Green
}

exit $codigo
