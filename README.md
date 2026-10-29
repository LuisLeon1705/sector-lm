# Sector-LM

Un experimento sobre modelos de lenguaje para código Python **divididos en sectores que se cargan bajo demanda**, en lugar de un único modelo que siempre está entero en memoria. Empezó como un sistema de n-gramas con tokens estructurados (`SS R C IIIIII`), Viterbi y tabla de símbolos, y terminó comparando expertos neuronales por dominio frente a modelos densos y a un MoE aprendido, todo medido con cuidado en una sola GPU de consumo (RTX 3070).

Los modelos neuronales de la ronda 5 forman la familia **Parallax**. El principal es **Parallax-15M**: un transformer de 15M de parámetros entrenado en unos 40 minutos que mejora al mejor n-grama por casi un factor de 2.

## Resultado en una tabla

PPC = perplejidad por carácter sobre los mismos 100 archivos de test (exactamente 795.700 caracteres). Menos es mejor.

| Modelo | Ronda | PPC | Notas |
| --- | :---: | ---: | --- |
| Sector-LM n-gramas, sectores sintácticos | 2 | 6,141 | 260 cargas de sector por cada 1.000 tokens |
| Sector-LM n-gramas, sectores por dominio | 2 | 4,698 | 0,4 cargas por cada 1.000 tokens |
| Sector-LM n-gramas, versión final (historia por sector) | 4 | 3,603 | 0,3 cargas por cada 1.000 tokens |
| N-grama BPE monolítico (comparación justa) | | 3,279 | |
| Parallax-E (E1): base de 15M + un adaptador LoRA por dominio | 5 | 1,724 | cambiar de dominio supone cargar ~4 MB |
| Parallax-M (M1): MoE Switch top-1, mismos parámetros | 5 | 1,725 | 2,7× más lento al inferir |
| Parallax-D 27M (D2): denso | 5 | 1,725 | |
| **Parallax-15M** (D1+): denso, 3 épocas | 5 | **1,718** | el modelo principal |
| `bigcode/tiny_starcoder_py` (164M, referencia) | | 1,296 | 11× más parámetros, ~600× más datos, ~7.000× más cómputo |

**Conclusión honesta:** enrutar por dominio ahorra mucha memoria, pero a esta escala **no mejora la calidad**. Los expertos por dominio, el MoE y el modelo denso grande empatan a menos del 1%, y entrenar más el modelo pequeño gana a todos. El límite son los datos, no la arquitectura.

## Documentación

| Documento | Contenido |
| --- | --- |
| [docs/00_idea_y_teoria.md](docs/00_idea_y_teoria.md) | La idea, los tokens estructurados, los sectores y cómo se mide |
| [docs/01_ronda1_prototipo.md](docs/01_ronda1_prototipo.md) | Primer prototipo y por qué su métrica no era justa |
| [docs/02_ronda2_precio_justo_y_dominios.md](docs/02_ronda2_precio_justo_y_dominios.md) | Precio justo para cada carácter y sectores por dominio |
| [docs/03_ronda3_unk_y_texto.md](docs/03_ronda3_unk_y_texto.md) | `<unk>` aprendido, BPE para comentarios y strings, y el error escondido |
| [docs/04_ronda4_historia.md](docs/04_ronda4_historia.md) | La misma historia al entrenar y al evaluar |
| [docs/05_ronda5_expertos_neuronales.md](docs/05_ronda5_expertos_neuronales.md) | Expertos neuronales por dominio frente a denso y MoE |
| [docs/06_velocidad_y_memoria.md](docs/06_velocidad_y_memoria.md) | Velocidad y memoria de todos los modelos, StarCoder incluido |
| [docs/07_conclusiones.md](docs/07_conclusiones.md) | Qué quedó demostrado, qué no y siguientes pasos |
| [docs/reproducir.md](docs/reproducir.md) | Comandos para reproducir cada ronda y probar los modelos |
| [docs/timeline.md](docs/timeline.md) | Cronología de todas las pruebas, hora a hora |
| [docs/referencias.md](docs/referencias.md) | Orígenes de las ideas: artículos y fichas, con las verificadas marcadas |

## Inicio rápido

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -r requirements.txt
cargo build --release
.venv\Scripts\python.exe scripts/prepare_data.py --train-limit 20000 --valid-limit 500 --test-limit 500
```

Después, cada ronda tiene sus comandos en [docs/reproducir.md](docs/reproducir.md). Cuando hayas entrenado la ronda 5, puedes probar un modelo:

```powershell
.venv\Scripts\python.exe -m neural.generate --prompt "def fibonacci(n):\n"
```

## Estructura

| Carpeta | Qué hay |
| --- | --- |
| `src/` | Sector-LM en Rust: parser, sectores, n-gramas Kneser-Ney, HMM, tabla de símbolos, almacenamiento y caché |
| `src/bin/` | `sector_eval` (evaluador de las rondas 1–4) y `dump_domains` (dominio de cada archivo) |
| `tests/` | Tests de Rust: equivalencia del top-k, reconstrucción del formato, evaluación paralela |
| `neural/` | Ronda 5 en PyTorch: modelos, entrenamiento, evaluación, generación y tests |
| `scripts/` | Preparación de datos, evaluación de StarCoder y benchmarks de velocidad |
| `results/` | Resultados medidos de cada ronda, en JSON |
| `data/` | Informes de datos y dominios (los datos en sí se descargan) |
