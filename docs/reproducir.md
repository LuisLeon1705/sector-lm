# Cómo reproducir los resultados

Todo se ejecuta desde la raíz del repositorio. Los resultados se midieron en Windows 11 con una RTX 3070 (8 GB), Rust 1.96 y Python 3.12.

## Entorno

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -r requirements.txt
cargo build --release
```

## Datos

Descarga en streaming los splits públicos de CodeParrot, descarta el Python inválido y evita cruces entre train y test por hash:

```powershell
.venv\Scripts\python.exe scripts/prepare_data.py --train-limit 20000 --valid-limit 500 --test-limit 500
```

Crea `data/train.jsonl`, `data/valid.jsonl`, `data/test.jsonl` y `data/leakage_report.json`. Todas las evaluaciones usan los primeros 100 archivos del test, exactamente 795.700 caracteres. Si la descarga devuelve archivos distintos, el evaluador lo detecta: comprueba ese número exacto.

Referencia `tiny_starcoder_py`:

```powershell
.venv\Scripts\python.exe scripts/eval_starcoder.py --test-file data/test.jsonl --max-files 100 --window-size 2048 --stride 1024 --output results/starcoder_metrics.json
```

## Rondas 1–4: n-gramas (Rust)

Tests:

```powershell
cargo test --release
```

El evaluador entrena, evalúa y compara con el n-grama BPE y con StarCoder. La configuración por defecto es la final de la ronda 4: sectores por dominio, `<unk>` aprendido, BPE de texto, ortografía por frecuencia e historia por sector.

```powershell
cargo run --release --bin sector_eval -- --output-summary results/mi_corrida.json --storage-dir sectors_storage
```

Variantes, cambiando una sola opción:

| Experimento | Opciones |
| --- | --- |
| Ronda 4 B, historia global | `--history mixed` |
| Sectores sintácticos | `--sectors syntactic` |
| Sin `<unk>` aprendido | `--unk-singletons false` |
| Sin BPE de texto | `--text-bpe false` |
| Ortografía sin ponderar | `--spelling-frequency false` |
| Ablaciones de la ronda 2 | `--price-sector`, `--format`, `--spelling` con `true` o `false` |

Las rondas 2 y 3 se midieron con versiones anteriores del código, antes del arreglo de la historia de la ronda 4, así que sus cifras exactas no se reproducen con el código actual. Sus resultados guardados están en `results/` y `results/round3/`.

## Ronda 5: modelos neuronales (Python + GPU)

```powershell
.venv\Scripts\python.exe -m neural.data --no-download
cargo run --release --bin dump_domains
.venv\Scripts\python.exe -m neural.pipeline --mode prepare
.venv\Scripts\python.exe -m neural.pipeline --mode smoke
.venv\Scripts\python.exe -m neural.pipeline --mode full
```

1. `neural.data` deduplica el train en `data/round5/train.jsonl`.
2. `dump_domains` asigna un dominio a cada archivo según sus `import`.
3. `prepare` entrena el tokenizador y codifica los datos.
4. `smoke` mide 2 minutos por modelo y estima el tiempo total.
5. `full` entrena D1, D1+, E1, D2 y M1 (unas 2,5 h en una RTX 3070) y luego los evalúa.

Si el entrenamiento se corta, el mismo comando retoma desde el último checkpoint. Para volver a evaluar sin entrenar:

```powershell
.venv\Scripts\python.exe -m neural.pipeline --mode eval
```

Tests:

```powershell
.venv\Scripts\python.exe -m pytest -q neural/tests_py
```

Velocidad y memoria:

```powershell
.venv\Scripts\python.exe scripts/bench_round5_speed.py
.venv\Scripts\python.exe scripts/bench_starcoder_speed.py
```

## Probar un modelo

```powershell
.venv\Scripts\python.exe -m neural.generate --prompt "def fibonacci(n):\n"
.venv\Scripts\python.exe -m neural.generate
.venv\Scripts\python.exe -m neural.generate --model E1 --domain WebBackend --prompt "from flask import Flask\n"
```

Sin `--prompt` abre una sesión interactiva: escribes código y terminas con una línea vacía. Opciones:

- `--model`: D1+ (por defecto), D1, D2, E1 o M1;
- `--max-new`: tokens a generar, 200 por defecto;
- `--temperature`: 0,7 por defecto; con 0 elige siempre el token más probable;
- `--top-k`: 40 por defecto;
- `--seed`: para repetir el mismo resultado.

Los checkpoints no están en el repositorio (cientos de MB). Hay que entrenarlos con la ronda 5.
