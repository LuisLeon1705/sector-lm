# Cronología

Todo el proyecto se hizo en dos días, el 28 y el 29 de octubre de 2026. El código lo escribieron Antigravity y Codex a partir de prompts, y cada ronda se revisó, se reprodujo y se corrigió antes de la siguiente. Las horas salen de las fechas de los archivos de `results/` y de los checkpoints, así que son aproximadas.

## 28 de octubre

| Hora | Etapa | Qué pasó |
| --- | --- | --- |
| ~15:00 | **Ronda 1**: prototipo | Antigravity construye Sector-LM en Rust: parser tree-sitter, tokens `SS R C IIIIII`, n-gramas Kneser-Ney por sector, Viterbi, tabla de símbolos, sectores en disco con caché LRU |
| 15:19 | Datos | Se preparan train, valid y test desde CodeParrot, sin cruces entre ellos |
| 15:47 | Referencia | Se evalúa `tiny_starcoder_py`: **PPC 1,296** |
| ~16:00–17:30 | Revisión de la ronda 1 | La primera ejecución completa no termina en 40 minutos: top-k exhaustivo, un archivo rezagado y una barra de progreso invisible. Codex toma el relevo y lo arregla con top-k exacto y parada temprana |
| 17:48–17:53 | **Ronda 2**: precio justo | Ablaciones: el PPC sin correcciones (2,088) era optimista. Con precio de sector, formato y ortografía queda en **6,141** |
| 17:59 | Ronda 2: dominios | Sectores por dominio: **4,698**, y las cargas bajan de 260 a **0,4 por cada 1.000 tokens** |
| 18:20 | **Ronda 3** | `<unk>` aprendido con singletons: **3,961** |
| 18:24–18:36 | Ronda 3 | BPE de comentarios y strings (4,082) y ortografía por frecuencia (4,183): **empeoran** |
| ~18:45 | Revisión de la ronda 3 | Se encuentra la causa: la historia al entrenar (por sector) y al evaluar (mezclada) no coincide. Una prueba rápida da 3,603 |
| 19:08–19:19 | **Ronda 4** | Historia por sector (**3,603**) y global con marcadores (3,648). Sigue por detrás del n-grama BPE (3,279) |
| 19:34–19:38 | **Ronda 5**: datos | Deduplicación del train (25 duplicados fuera) y tokenizador BPE de 16.384 entradas |
| ~19:50 | Preflight 1 | Revisión: el router de M1 era falso y lento (5.172 tokens/s), y E1 tenía media base. Estimación: 7,85 h |
| 20:16–20:34 | Preflights corregidos | Router Switch real, expertos vectorizados (M1 a 61.896 tokens/s), E1 parte de D1, control D1+. Estimación: **1,97 h** |
| ~20:40–23:12 | Entrenamiento completo 1 | Termina, pero la evaluación falla al contar caracteres (795.825 en vez de 795.700) |
| 23:14 | Evaluación 1 | Arreglado el conteo, D1 da **4,155**, peor que el n-grama. Diagnóstico: embeddings iniciados en N(0, 1) y 256 `<bos>` nunca vistos |
| 23:22 | Prueba corta | Con la inicialización y el `<bos>` corregidos, 3 minutos de D1 ya ganan a los 26 minutos del D1 roto |

## 29 de octubre

| Hora | Etapa | Qué pasó |
| --- | --- | --- |
| ~09:45 | Entrenamiento completo 2 | Se lanzan los cinco modelos desde cero |
| 10:13 | D1 | Terminado (unos 26 min) |
| 10:27 | D1+ | Terminado |
| 10:40 | E1 | Terminados los 12 adaptadores |
| 11:13 | D2 | Terminado |
| ~11:23 | **Corte de luz** | Se corta en mitad del entrenamiento de M1, mientras se escribía su checkpoint, que queda corrupto. Los demás modelos están intactos |
| 11:30 | Reanudación | Se aparta el `M1.pt` corrupto, el guardado pasa a ser atómico y se retoma: los modelos terminados se saltan |
| 12:29–12:30 | **Resultados de la ronda 5** | M1 termina (58 min) y se evalúa todo: D1 1,728, **D1+ 1,718**, D2 1,725, E1 1,724, M1 1,725 |
| Tarde | Medición de velocidad y memoria | Cada modelo por separado con la GPU limpia, StarCoder incluido ([06_velocidad_y_memoria.md](06_velocidad_y_memoria.md)) |
| Tarde | Uso y documentación | `neural/generate.py` para probar los modelos, `.gitignore` y esta documentación |

## En resumen

```
Ronda 1  prototipo n-gramas ............ no terminaba; métrica injusta
Ronda 2  precio justo .................. 2,088 (falso) → 6,141
         sectores por dominio .......... 4,698   (cargas 260 → 0,4)
Ronda 3  <unk> aprendido ............... 3,961
         BPE de texto + ortografía ..... 4,183   (error escondido)
Ronda 4  historia corregida ............ 3,603   (n-grama BPE: 3,279)
Ronda 5  transformers de 15–27M ........ 1,718 – 1,728  (StarCoder: 1,296)
```
