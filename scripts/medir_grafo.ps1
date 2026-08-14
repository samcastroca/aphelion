<#
.SYNOPSIS
    Mide el grafo de conocimiento (§7, bonus) sobre una muestra y extrapola al corpus.

.DESCRIPTION
    Construir el grafo sobre los 64.484 fragmentos no es algo que convenga lanzar
    a ciegas: la primera version de la etapa tardaba 15 horas porque el NER corria
    en CPU y sin lotes. Este script mide primero sobre una muestra, extrapola, y
    solo corre el corpus entero si se lo pides con -Completo.

    **Por que existe en vez de una lista de comandos.** Los extras de uv tienen que
    ir en *todas* las llamadas, no solo en el sync: `uv run` sincroniza contra el
    conjunto que se le pida, asi que uno sin --extra cuda desinstala el torch de
    CUDA que el sync acaba de instalar y la etapa vuelve a correr en CPU sin que
    nada avise. Es la misma trampa que documenta ejecutar.ps1, y ya mordio una vez.
    Aqui los extras se componen una sola vez y viajan solos.

.PARAMETER Fragmentos
    Tamano de la muestra. 1000 da una medida estable en menos de un minuto.

.PARAMETER Lote
    Ventanas por lote de inferencia. Bajalo a 16 si la GPU se queda sin memoria;
    subelo a 64 si se queda corta de trabajo.

.PARAMETER Ner
    Backend: gliner2 (PyTorch+GPU, por defecto), onnx (DirectML en Radeon) o
    falso (sin modelo, para comprobar el camino entero sin gastar GPU).

.PARAMETER Completo
    Tras medir, construye el grafo sobre el corpus entero y lo deja donde la §1.4
    lo pide. Sin esto el script no toca entrega/.

.PARAMETER SinSync
    Salta la instalacion de dependencias. Para repetir medidas sin esperar a uv.

.EXAMPLE
    .\scripts\medir_grafo.ps1
    .\scripts\medir_grafo.ps1 -Fragmentos 2000 -Lote 64
    .\scripts\medir_grafo.ps1 -Ner falso          # sin GPU, comprueba el camino
    .\scripts\medir_grafo.ps1 -Completo           # ya medido: el corpus entero
#>

[CmdletBinding()]
param(
    [int]$Fragmentos = 1000,
    [int]$Lote = 32,
    [ValidateSet("gliner2", "onnx", "falso")]
    [string]$Ner = "gliner2",
    [switch]$Completo,
    [switch]$SinSync
)

$ErrorActionPreference = "Stop"

function Paso($texto)  { Write-Host "`n=== $texto ===" -ForegroundColor Cyan }
function Bien($texto)  { Write-Host "  $texto" -ForegroundColor Green }
function Aviso($texto) { Write-Host "  $texto" -ForegroundColor Yellow }
function Mal($texto)   { Write-Host "  $texto" -ForegroundColor Red }

# El script vive en scripts/, el proyecto es el directorio de arriba. Se ancla asi
# y no al directorio actual para que funcione llamandolo desde donde sea.
$raiz = Split-Path -Parent $PSScriptRoot
Push-Location $raiz
try {
    if (-not (Test-Path "pyproject.toml")) {
        throw "no encuentro pyproject.toml en $raiz; este script va dentro del repo"
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv no esta en el PATH. Instalalo: https://docs.astral.sh/uv/"
    }

    # --- Extras -----------------------------------------------------------
    # Se componen una vez y viajan en todas las llamadas a uv. Ver el bloque de
    # .DESCRIPTION: sin esto, cada `uv run` deshace el sync anterior.
    Paso "Entorno"
    $gpu = @(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name)
    $nvidia = @($gpu | Where-Object { $_ -match "NVIDIA|GeForce|RTX" }).Count -gt 0
    $radeon = @($gpu | Where-Object { $_ -match "Radeon|AMD" }).Count -gt 0

    $extras = @()
    if ($nvidia) {
        $extras += @("--extra", "cuda")
        Bien "GPU NVIDIA detectada"
    } elseif ($radeon) {
        $extras += @("--extra", "amd")
        Bien "GPU Radeon detectada"
    } else {
        Aviso "sin GPU reconocida: el NER correra en CPU y tardara horas"
    }
    $extras += @("--extra", "grafo")
    Write-Host "  extras: $($extras -join ' ')"

    if ($SinSync) {
        Aviso "-SinSync: no se tocan las dependencias"
    } else {
        Paso "Dependencias"
        & uv sync @extras
        if ($LASTEXITCODE -ne 0) { throw "uv sync fallo" }
        Bien "sincronizado"
    }

    # --- Calentamiento ----------------------------------------------------
    # Incondicional, y antes de cualquier cronometro. `uv run` sincroniza el
    # entorno contra los extras que se le pasen, asi que la *primera* llamada con
    # extras nuevos instala paquetes aunque el `uv sync` de arriba se haya saltado
    # con -SinSync. Medido: 15 s de instalacion colandose dentro del reloj de la
    # muestra, que es medir el instalador y llamarlo NER.
    Paso "Entorno de ejecucion"
    & uv run @extras python -c "pass"
    if ($LASTEXITCODE -ne 0) { throw "el entorno no arranca" }
    Bien "listo"

    # --- Comprobacion de GPU ----------------------------------------------
    # Antes de gastar nada. Un torch sin CUDA no falla: corre en CPU, tarda 15
    # horas y solo se nota al final, que es exactamente como llegamos aqui.
    if ($Ner -eq "gliner2") {
        Paso "PyTorch ve la GPU"
        $sonda = 'import torch; print("TORCH", torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")'
        $linea = @(& uv run @extras python -c $sonda) | Where-Object { $_ -like "TORCH *" } | Select-Object -First 1
        if ($LASTEXITCODE -ne 0 -or -not $linea) { throw "no pude interrogar a torch" }

        Write-Host "  $linea"
        $campos = $linea -split "\s+"
        if ($campos[3] -ne "True") {
            Mal "torch no ve la GPU: la corrida seria en CPU y son horas."
            Mal "Si la version no acaba en +cu130, el extra no se aplico:"
            Mal "    uv sync --extra cuda --extra grafo"
            Mal "y recuerda que todo `uv run` necesita esos mismos --extra."
            throw "sin GPU no tiene sentido medir"
        }
        Bien "$($campos[4])"

        # --- Descarga del modelo ------------------------------------------
        # Aparte de la medicion: son ~800 MB la primera vez y falsearian el reloj.
        Paso "Modelo de NER"
        & uv run @extras python scripts/etapas/04_grafo.py --limite 20 --rehacer-ner `
            --ner $Ner --lote $Lote --destino "trabajo/descarte.graphml" | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "no pude cargar el modelo de NER" }
        Bien "listo en cache"
    }

    # --- Medicion ---------------------------------------------------------
    Paso "Muestra de $Fragmentos fragmentos"
    $crono = [System.Diagnostics.Stopwatch]::StartNew()
    & uv run @extras python scripts/etapas/04_grafo.py `
        --limite $Fragmentos --rehacer-ner --ner $Ner --lote $Lote `
        --destino "trabajo/grafo_muestra.graphml" | Tee-Object -Variable salida
    $crono.Stop()
    if ($LASTEXITCODE -ne 0) { throw "la corrida de muestra fallo" }

    $segundos = $crono.Elapsed.TotalSeconds

    # El total sale del propio corpus y no de una constante: una submuestra o un
    # corpus recortado darian una extrapolacion falsa sin que nada lo dijera.
    $total = [int](& uv run @extras python scripts/etapas/04_grafo.py --contar)
    if ($LASTEXITCODE -ne 0) { throw "no pude contar los fragmentos del corpus" }

    $medidos = [Math]::Min($Fragmentos, $total)
    $factor = $total / [double]$medidos
    $estimado = ($segundos * $factor) / 60.0

    Paso "Resultado"
    Write-Host ("  muestra          {0:N0} fragmentos en {1:N1} s" -f $medidos, $segundos)
    Write-Host ("  ritmo            {0:N1} fragmentos/s" -f ($medidos / $segundos))
    Write-Host ("  corpus           {0:N0} fragmentos (x{1:N1})" -f $total, $factor)
    Write-Host ("  ESTIMADO TOTAL   {0:N1} min" -f $estimado) -ForegroundColor Cyan

    # Lo que el NER descarta por offsets que no cuadran no rompe nada, pero vacia
    # la evidencia de las tripletas: si el numero es grande, el modelo devuelve
    # posiciones en otra unidad y hay que mirarlo antes de fiarse del grafo.
    $descartes = @($salida | Where-Object { $_ -match "offsets que no cuadran" })
    if ($descartes.Count -gt 0) {
        Aviso "revisa: $($descartes[0])"
    }

    if ($estimado -gt 60) {
        Aviso "mas de una hora: algo no esta usando la GPU. No lances -Completo todavia."
    } else {
        Bien "dentro de lo razonable"
    }

    if (-not $Completo) {
        Write-Host "`n  Para el corpus entero, cuando el numero convenza:" -ForegroundColor DarkGray
        Write-Host "      .\scripts\medir_grafo.ps1 -Completo -SinSync" -ForegroundColor DarkGray
        return
    }

    # --- Corpus entero ----------------------------------------------------
    # --rehacer-ner no es opcional: sin el, esta corrida leeria la cache que acaba
    # de dejar la muestra y construiria el grafo sobre esos $Fragmentos.
    Paso "Corpus entero"
    $crono = [System.Diagnostics.Stopwatch]::StartNew()
    & uv run @extras python scripts/etapas/04_grafo.py --rehacer-ner --ner $Ner --lote $Lote
    $crono.Stop()
    if ($LASTEXITCODE -ne 0) { throw "la corrida completa fallo" }

    Paso "Hecho"
    Write-Host ("  tiempo real      {0:N1} min (estimado {1:N1})" -f $crono.Elapsed.TotalMinutes, $estimado)
    Bien "grafo en entrega/base_vectorial/grafo/grafo.graphml"
}
finally {
    Pop-Location
}
