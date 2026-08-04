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

.PARAMETER Submuestra
    Elige el subconjunto de documentos con el que iterar rápido y sale. Mantiene
    dentro todo lo que el ground truth toca, añade los negativos que los encoders
    puntúan alto sin ser relevantes, y completa con un muestreo estratificado.
    Escribe data\submuestra.json, que es lo que consume el barrido.

    Exige tener anotado el ground truth y el índice de referencia construido.

    Que el ground truth entre entero no es una preferencia: si un documento
    juzgado no fuera alcanzable, el guion falla y no escribe la submuestra, en
    vez de dejar que el barrido mida más bajo sin que nada lo indique.

.PARAMETER Barrido
    Corre un experimento sobre la submuestra con las opciones que se le pidan y
    guarda sus métricas en pruebas\<nombre>\. No prueba todas las combinaciones:
    cada dimensión se elige por separado, y lo que no se diga toma su valor de la
    entrega. Sin más parámetros el menú pregunta cada cosa.

.PARAMETER Recetas
    Configuraciones **completas** a correr, por nombre y separadas por comas, o
    "todas". Cada receta trae ya todos sus parámetros —encoders, tokens, solape,
    fusión, agregación, realce, tope por documento, profundidad y umbral— junto
    con la apuesta que hace y en qué se basa. Cada una es una corrida, así que la
    tabla que sale se lee de arriba abajo sin descontar sobreajuste, y todas se
    comparan contra la receta `entrega`, que es lo que está construido hoy.

      entrega      la línea base: lo que se entrega hoy
      dos-encoders lo que se entregaba antes: bge-m3 + mE5-large por RRF
      convexa      fusionar por similitud normalizada en vez de por posición
      familias     fusionar dos familias distintas en vez de dos parientes
      filtrado     descartar la cola floja de cada índice antes de fusionar
      barato       ¿hace falta un modelo grande?
      sin-recorte  fragmentos que quepan enteros en las 250 palabras del reto
      granular     fragmentos cortos y sin solape, apostando al NDCG@10
      contexto     fragmentos largos y top3, apostando al F1@3

    El catálogo con sus parámetros y sus motivos:
        uv run python -m aphelion.evaluacion.recetas

.PARAMETER Preset
    Una selección ya hecha, para las preguntas que suelen interesar. Abre una
    dimensión y deja el resto fija, que es como se lee un resultado sin confundir
    el efecto de una cosa con el de otra. A diferencia de una receta, no fija
    todos los parámetros: sirve para entender una dimensión, no para elegir qué
    entregar.

      rapido      Valida la maquinaria en minutos. No decide nada.
      chunking    Tamaño de chunk y solape, con el encoder más barato. El primer
                  paso: descarta media rejilla por una fracción del coste.
      encoders    Qué encoder solo y qué pares, sobre el chunking ya elegido.
      fusion      RRF frente a CombSUM frente a la convexa normalizada, y k0.
      agregacion  Max, suma, media y top-N para pasar de fragmentos a documentos.
      politicas   Todo lo barato, con encoders y chunking fijos.
      todo        El catálogo entero. Es una noche de GPU.

.PARAMETER Nombre
    Nombre del experimento, que da nombre a su carpeta en pruebas\. Por defecto
    el preset más la fecha.

.PARAMETER Chunks
    Tokens por fragmento a probar, uno o varios: -Chunks 256,504,768

.PARAMETER Solapes
    Fracciones de solape a probar: -Solapes 0,0.15

.PARAMETER Fusiones
    Cómo combinar dos espacios vectoriales: rrf, combsum, convexa.

.PARAMETER Agregaciones
    Cómo pasar de fragmentos a documentos: max, suma, media, top2, top3.

.PARAMETER MaxFusion
    Cuántos encoders puede fusionar una corrida a la vez. 1 los prueba solo por
    separado; 2 añade los pares. Por defecto 2.

.PARAMETER ListarPruebas
    Muestra lo mejor de cada experimento ya corrido y sale.

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

    # Experimentación: primero la submuestra, después los experimentos
    .\ejecutar.ps1 -Submuestra
    .\ejecutar.ps1 -Barrido -Recetas entrega,dos-encoders  # la comparación que decide
    .\ejecutar.ps1 -Barrido -Recetas todas
    .\ejecutar.ps1 -Barrido -Preset chunking
    .\ejecutar.ps1 -Barrido -Encoders bge-m3,gte-multilingual-base -Fusiones rrf,convexa
    .\ejecutar.ps1 -Barrido -Nombre solo-bge -Encoders bge-m3 -MaxFusion 1
    .\ejecutar.ps1 -ListarPruebas             # comparar lo ya corrido

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
    [switch]$Auto,
    [switch]$Submuestra,
    [switch]$Barrido,
    # Sin ValidateSet: el catálogo de recetas vive en Python y duplicar los
    # nombres aquí haría que añadir una receta dejara el guion rechazándola.
    [string[]]$Recetas,
    [ValidateSet('rapido', 'chunking', 'encoders', 'fusion', 'agregacion', 'politicas', 'todo')]
    [string]$Preset,
    [string]$Nombre,
    [int[]]$Chunks,
    [double[]]$Solapes,
    [ValidateSet('rrf', 'combsum', 'convexa')]
    [string[]]$Fusiones,
    [ValidateSet('max', 'suma', 'media', 'top2', 'top3')]
    [string[]]$Agregaciones,
    [ValidateRange(1, 4)]
    [int]$MaxFusion,
    [switch]$ListarPruebas
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Paso($texto) { Write-Host "`n=== $texto ===" -ForegroundColor Cyan }
function Aviso($texto) { Write-Host "  ! $texto" -ForegroundColor Yellow }
function Bien($texto) { Write-Host "  $texto" -ForegroundColor Green }

# Extras de uv que le corresponden a esta máquina. Se rellena al detectar la GPU
# y viaja en *todas* las llamadas a uv, no solo en el sync: `uv run` sincroniza
# contra el conjunto que se le pida, así que un `uv run` sin --extra cuda
# desinstala el torch de CUDA que el sync acababa de instalar.
$script:Extras = @()

function CorrerUv {
    <#
        Llama a uv sin que su salida por stderr mate el guion.

        Windows PowerShell 5.1 convierte en error terminante cualquier escritura
        a stderr de un ejecutable cuando $ErrorActionPreference es Stop, y uv
        informa por ahí de lo que instala y desinstala. Quien decide si hubo
        fallo es el código de salida, y eso es lo que se mira después.

        Asignar el resultado captura la salida; no asignarlo la deja pasar en
        directo, que es lo que quiere el pipeline.
    #>
    param([Parameter(ValueFromRemainingArguments = $true)]$argumentos)
    $previo = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        # Por la ruta del ejecutable: una función llamada 'uv' se llamaría a sí
        # misma, porque PowerShell no distingue mayúsculas en los nombres.
        & (Get-Command uv -CommandType Application | Select-Object -First 1).Source @argumentos 2>&1 |
            ForEach-Object { "$_" }
    } finally {
        $ErrorActionPreference = $previo
    }
}

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

function ElegirVarias($titulo, [string[]]$etiquetas, [string[]]$valores, $porDefecto,
                      [string[]]$notas = @()) {
    <#
        Selección múltiple: "1,3" toma la primera y la tercera, "t" toma todas.

        Devuelve los valores elegidos, no sus índices, para que quien llama no
        tenga que volver a mapear. Un experimento se define eligiendo varias
        opciones de varias dimensiones, y preguntarlas de una en una obligaría a
        correr el guion tantas veces como valores se quieran probar.

        `notas` añade una segunda línea a cada opción. La usan las recetas, donde
        la etiqueta son los parámetros y la nota es la apuesta que hacen: elegir
        entre nueve configuraciones sin saber qué apuesta cada una no es elegir.
    #>
    Paso $titulo
    for ($i = 0; $i -lt $etiquetas.Count; $i++) {
        Write-Host ("  {0}) {1}" -f ($i + 1), $etiquetas[$i])
        if ($i -lt $notas.Count -and $notas[$i]) {
            Write-Host ("       {0}" -f $notas[$i]) -ForegroundColor DarkGray
        }
    }
    Write-Host "  t) todas"
    while ($true) {
        $r = Leer "Elige (varias con coma)" $porDefecto
        if ($r -eq 't') { return $valores }
        $piezas = $r -split '[,\s]+' | Where-Object { $_ }
        $malas = $piezas | Where-Object {
            $_ -notmatch '^\d+$' -or [int]$_ -lt 1 -or [int]$_ -gt $etiquetas.Count
        }
        if (-not $malas -and $piezas.Count) {
            # Sin duplicados y en el orden en que están listadas, que es el orden
            # de coste creciente: así una corrida interrumpida deja lo barato hecho.
            $indices = $piezas | ForEach-Object { [int]$_ - 1 } | Select-Object -Unique | Sort-Object
            return @($indices | ForEach-Object { $valores[$_] })
        }
        Aviso "Números entre 1 y $($etiquetas.Count), separados por coma. O 't'."
    }
}

function LeerRecetas {
    <#
        El catálogo de recetas, tal como lo declara Python.

        Se lee y no se copia: si las etiquetas estuvieran escritas aquí, añadir
        una receta o cambiarle un parámetro dejaría el menú describiendo algo que
        ya no es. Cada línea trae nombre, configuración, apuesta y coste.

        Va por el intérprete del entorno y no por `uv run`, que sincroniza contra
        los extras que se le pasen: uno sin --extra cuda desinstalaría el torch de
        CUDA en medio del menú. Si el entorno no está preparado todavía devuelve
        vacío, y quien llama decide qué hacer.
    #>
    $python = ".venv\Scripts\python.exe"
    if (-not (Test-Path $python)) { return @() }
    $previo = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $python -m aphelion.evaluacion.recetas --tsv 2>$null |
            Where-Object { $_ -match "`t" }
    } finally {
        $ErrorActionPreference = $previo
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
Bien (CorrerUv --version)

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
            "Experimentar: submuestra y barrido de configuraciones"
        )) {
        5 {
            # La submuestra y los experimentos son el ciclo de mejora, no la
            # entrega: se apoyan en el ground truth anotado y en el índice de
            # referencia, y no tocan ni entrega\ ni resultados.jsonl.
            switch (Elegir "Qué parte del ciclo" @(
                    "Elegir la submuestra de documentos       (primero esto)"
                    "Correr recetas completas                 (lo normal: candidatas a entregar)"
                    "Configurar un experimento a medida       (elegir cada opción)"
                    "Abrir una sola dimensión                 (preset: para entenderla)"
                    "Ver los experimentos ya corridos"
                    "Validar la maquinaria en minutos"
                ) 2) {
                1 { $Submuestra = [switch]$true }
                2 {
                    $Barrido = [switch]$true

                    $catalogo = @(LeerRecetas)
                    if ($catalogo.Count) {
                        $nombres = @(); $etiquetas = @(); $apuestas = @()
                        foreach ($fila in $catalogo) {
                            $campos = $fila -split "`t"
                            $nombres += $campos[0]
                            $etiquetas += ("{0,-13} {1}" -f $campos[0], $campos[1])
                            $apuestas += ("{0}  ({1})" -f $campos[2], $campos[3])
                        }
                        # Por defecto la entrega y la candidata con mejor indicio
                        # medido: es la comparación que decide si el segundo
                        # encoder sigue justificándose.
                        $Recetas = ElegirVarias `
                            "Qué recetas correr (cada una es una corrida)" `
                            $etiquetas $nombres "1,2" $apuestas
                    } else {
                        Aviso "El catálogo se lee del entorno, que aún no está preparado."
                        Aviso "Se correrán 'entrega' y 'dos-encoders'; para verlo todo:"
                        Aviso "  uv run python -m aphelion.evaluacion.recetas"
                        $Recetas = @("entrega", "dos-encoders")
                    }

                    # Windows PowerShell 5.1 no tiene operador ternario.
                    $base = if ($Recetas.Count -eq 1) { $Recetas[0] } else { "recetas" }
                    $Nombre = Leer "Nombre del experimento" (
                        "{0}-{1}" -f $base, (Get-Date -Format "HHmm")
                    )
                }
                3 {
                    $Barrido = [switch]$true

                    # Las etiquetas llevan por qué está cada opción: elegir a
                    # ciegas entre siete encoders no es elegir.
                    $Encoders = ElegirVarias "Qué encoders probar" @(
                        "me5-small              384d  el más barato del catálogo"
                        "me5-base               768d  barato y más capaz"
                        "bge-m3                1024d  el principal de la entrega"
                        "me5-large             1024d  el complementario, retirado tras medirlo"
                        "gte-multilingual-base  768d  familia distinta, más consenso nuevo"
                        "me5-large-instruct    1024d  asimetría instruida, para es->en"
                        "labse                  768d  control: el informe lo descarta"
                    ) @(
                        "me5-small", "me5-base", "bge-m3", "me5-large",
                        "gte-multilingual-base", "me5-large-instruct", "labse"
                    ) "3,4"

                    if ($Encoders.Count -gt 1) {
                        $MaxFusion = switch (Elegir "Cómo probarlos" @(
                                "Solos y por pares   (mide qué aporta la fusión)"
                                "Solo por separado   (más rápido)"
                            )) {
                            2 { 1 }
                            default { 2 }
                        }
                    } else {
                        $MaxFusion = 1
                    }

                    $Chunks = [int[]](ElegirVarias "Tokens por fragmento" @(
                        "256   más granularidad, favorece al NDCG@10"
                        "504   el de la entrega"
                        "768   más contexto, favorece al F1@3"
                    ) @("256", "504", "768") "2")

                    $Solapes = [double[]](ElegirVarias "Solape entre fragmentos" @(
                        "0      sin solape; hay evidencia de que no aporta"
                        "0.15   el de la entrega, cuesta un 15% de vectores"
                    ) @("0", "0.15") "2")

                    $Fusiones = ElegirVarias "Cómo fusionar los rankings" @(
                        "rrf       por posiciones; el de la entrega"
                        "combsum   suma las similitudes crudas"
                        "convexa   suma las similitudes normalizadas"
                    ) @("rrf", "combsum", "convexa") "1,3"

                    $Agregaciones = ElegirVarias "Cómo pasar de fragmentos a documentos" @(
                        "max     el mejor fragmento manda; el de la entrega"
                        "suma    acumula; tiene sesgo de longitud"
                        "media   sin sesgo de longitud, castiga al documento largo"
                        "top2    media de los dos mejores"
                        "top3    media de los tres mejores"
                    ) @("max", "suma", "media", "top2", "top3") "1,4"

                    $Nombre = Leer "Nombre del experimento" (
                        "{0}-{1}" -f ($Encoders -join "+"), (Get-Date -Format "HHmm")
                    )
                }
                4 {
                    $Barrido = [switch]$true
                    $Preset = switch (Elegir "Qué dimensión abrir" @(
                            "chunking     tamaño de chunk y solape, encoder barato"
                            "encoders     qué encoder solo y qué pares"
                            "fusion       RRF vs CombSUM vs convexa, y k0"
                            "agregacion   max, suma, media, top-N"
                            "politicas    todo lo barato, encoders fijos"
                            "todo         el catálogo entero (una noche de GPU)"
                        )) {
                        1 { "chunking" }
                        2 { "encoders" }
                        3 { "fusion" }
                        4 { "agregacion" }
                        5 { "politicas" }
                        6 { "todo" }
                    }
                }
                5 { $ListarPruebas = [switch]$true }
                6 { $Barrido = [switch]$true; $Preset = "rapido" }
            }
        }
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

    # La submuestra y el barrido eligen sus propios encoders —el barrido los
    # barre, que es el punto— así que preguntar aquí no tendría sentido.
    if (-not $SoloEntorno -and -not $Submuestra -and -not $Barrido) {
        $Encoders = switch (Elegir "Qué encoders indexar" @(
                "Solo bge-m3                   (lo que se entrega)"
                "Los dos: bge-m3 y me5-large   (la entrega anterior, el doble de tiempo)"
                "Solo me5-large"
            )) {
            1 { @("bge-m3") }
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

# Listar los experimentos ya corridos no necesita entorno, ni GPU, ni corpus: son
# archivos en pruebas\. Se atiende antes de todo lo demás.
if ($ListarPruebas) {
    Paso "Experimentos corridos"
    & (Get-Command uv -CommandType Application | Select-Object -First 1).Source `
        run python scripts/analisis/barrido_completo.py --listar
    exit $LASTEXITCODE
}

# Pedir recetas o un preset es pedir un experimento. Sin esto, `-Recetas todas` a
# secas se pondría a construir la entrega, que es lo contrario de lo que se pidió.
if (($Recetas -or $Preset) -and -not $Barrido) { $Barrido = [switch]$true }

$experimento = $Submuestra -or $Barrido

if ($experimento) {
    # El ciclo de experimentación no parte del corpus sino del texto ya
    # extraído y del ground truth anotado. No toca la entrega.
    Paso "Insumos del experimento"
    $faltan = @()
    if (-not (Test-Path "trabajo\texto")) { $faltan += "trabajo\texto\ (corre antes 01_extraer)" }
    if (-not (Test-Path "data\ground_truth.jsonl")) { $faltan += "data\ground_truth.jsonl" }
    if ($Barrido -and -not (Test-Path "data\ground_truth_textos.jsonl")) {
        $faltan += "data\ground_truth_textos.jsonl (pool_juicios.py --consolidar)"
    }
    if ($Barrido -and -not (Test-Path "data\submuestra.json")) {
        $faltan += "data\submuestra.json (.\ejecutar.ps1 -Submuestra)"
    }
    if ($faltan.Count) {
        Write-Host "  Faltan insumos:" -ForegroundColor Red
        $faltan | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
        exit 1
    }
    $n = (Get-ChildItem "trabajo\texto" -File | Measure-Object).Count
    Bien "$n documentos extraídos, ground truth presente"
} elseif ($Reparto) {
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
# El extra decide de qué índice sale torch, y tiene que ir en todas las llamadas
# a uv: `uv run` sincroniza contra lo que se le pida, así que uno sin --extra
# cuda desinstala el torch de CUDA que el sync acaba de poner.
if ($nvidia) {
    $script:Extras = @("--extra", "cuda")
} elseif ($radeon) {
    $script:Extras = @("--extra", "amd")
}

CorrerUv sync @script:Extras
if ($LASTEXITCODE -ne 0) { throw "uv sync falló" }

if ($nvidia) {
    $torch = CorrerUv run @script:Extras python -c "import torch; print('GPU', torch.cuda.is_available(), torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    $linea = ($torch | Where-Object { $_ -like 'GPU *' } | Select-Object -First 1)
    if ($linea -like 'GPU True*') {
        Bien ($linea -replace '^GPU True ', 'torch ')
    } else {
        Aviso "PyTorch no ve la GPU pese a la build CUDA. Revisa el driver."
    }
}

# Con reparto o experimentando no se extrae nada, así que el OCR no entra en juego.
if (-not $Reparto -and -not $experimento) {
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

if ($Submuestra) {
    Paso "Submuestra"
    CorrerUv (@("run") + $script:Extras + @("python", "scripts/analisis/submuestra.py"))
    exit $LASTEXITCODE
}

if ($Barrido) {
    Paso "Experimento"
    $argumentos = @("run") + $script:Extras +
        @("python", "scripts/analisis/barrido_completo.py")

    if ($Recetas) { $argumentos += @("--recetas", ($Recetas -join ",")) }
    if ($Preset) { $argumentos += @("--preset", $Preset) }
    if ($Nombre) { $argumentos += @("--nombre", $Nombre) }
    if ($Encoders) { $argumentos += @("--encoders", ($Encoders -join ",")) }
    if ($Chunks) { $argumentos += @("--chunks", ($Chunks -join ",")) }
    if ($Solapes) {
        # La coma decimal de una configuración regional en español no la entiende
        # Python: 0,15 llegaría partido en dos valores. Se fuerza el punto.
        $texto = ($Solapes | ForEach-Object {
                $_.ToString([cultureinfo]::InvariantCulture)
            }) -join ","
        $argumentos += @("--solapes", $texto)
    }
    if ($Fusiones) { $argumentos += @("--fusiones", ($Fusiones -join ",")) }
    if ($Agregaciones) { $argumentos += @("--agregaciones", ($Agregaciones -join ",")) }
    if ($MaxFusion) { $argumentos += @("--max-fusion", $MaxFusion) }

    # El backend y el lote los decide el mismo hardware que en el pipeline: sin
    # esto, un experimento en la Radeon iría por CPU y tardaría días.
    if ($Backend) {
        $argumentos += @("--backend", $Backend)
    } elseif ($radeon) {
        $argumentos += @("--backend", "onnx")
    }
    if ($Lote) {
        $argumentos += @("--lote", $Lote)
    } elseif ($radeon) {
        $argumentos += @("--lote", "8")
    } elseif ($nvidia) {
        $argumentos += @("--lote", "128")
    }

    CorrerUv @argumentos
    $codigo = $LASTEXITCODE
    if ($codigo -eq 0) {
        Write-Host ""
        Write-Host "  Compáralo con los anteriores: .\ejecutar.ps1 -ListarPruebas" -ForegroundColor Green
    }
    exit $codigo
}

Paso "Pipeline"
# Los nombres de encoder no se validan aquí: los define config.ENCODERS y es
# pipeline.py quien los comprueba, antes de tocar nada.
$argumentos = @("run") + $script:Extras + @("python", "scripts/pipeline.py")
if ($Desde) { $argumentos += @("--desde", $Desde) }
if ($Encoders) { $argumentos += @("--encoders", ($Encoders -join ",")) }
if ($Backend) { $argumentos += @("--backend", $Backend) }
if ($Lote) { $argumentos += @("--lote", $Lote) }
if ($SinOcr) { $argumentos += "--sin-ocr" }
if ($Forzar) { $argumentos += "--forzar" }
if ($Reparto) { $argumentos += @("--reparto", $Reparto) }

CorrerUv @argumentos
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
