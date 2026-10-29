# Ronda 5: expertos neuronales por dominio

## Hipótesis

Los n-gramas tocaron techo. La pregunta pasa a ser si la idea de Sector-LM aporta algo con modelos neuronales: un **modelo base compartido más un adaptador por dominio**, enrutado por los `import` y cargado bajo demanda. ¿Es mejor que un modelo denso o que un MoE aprendido con el mismo presupuesto?

## Datos

- **Train**: `data/round5/train.jsonl`, 19.975 archivos tras quitar 25 duplicados. Se normalizó a NFC y saltos de línea LF antes de comparar hashes, y hay cero cruces con valid y test (`data/leakage_report.json`).
- **Tokenizador**: un único BPE byte-level de 16.384 entradas, compartido por todos los modelos: **53.410.402 tokens** de train.
- **Test**: los mismos 100 archivos y 795.700 caracteres de todas las rondas.
- **Dominios**: el clasificador de Rust (`src/bin/dump_domains.rs`) mira solo los `import`. Los dominios con menos del 3% de los tokens se juntan con GeneralLocal (solo AsyncConcurrency), y quedan 12 expertos:

| Dominio | Tokens de train | % |
| --- | ---: | ---: |
| SystemOsCli | 7.427.594 | 13,91 |
| WebBackend | 7.065.169 | 13,23 |
| GeneralLocal (+ AsyncConcurrency) | 6.552.542 | 12,27 |
| DataStorageFormats | 6.276.757 | 11,75 |
| ScientificNumeric | 5.421.856 | 10,15 |
| NetworkingCloud | 4.816.984 | 9,02 |
| TestingQa | 3.900.171 | 7,30 |
| StdlibCoreTypes | 3.655.927 | 6,84 |
| MachineLearning | 2.494.658 | 4,67 |
| TextCryptoMath | 2.359.315 | 4,42 |
| RuntimeInspectPkg | 1.776.605 | 3,33 |
| GuiGraphicsMedia | 1.662.824 | 3,11 |

## Los modelos

Todos son decoders tipo GPT con 4 bloques, contexto de 512 y embeddings atados a la capa de salida (`neural/model.py`):

| Nombre | Qué es | Parámetros totales | Activos por token |
| --- | --- | ---: | ---: |
| **D1** | Denso pequeño (ancho 368) | 14.891.488 | 14.891.488 |
| **D1+** | D1 continuado una época más con todos los datos | 14.891.488 | 14.891.488 |
| **D2** | Denso con tantos parámetros totales como E1 y M1 | 26.772.768 | 26.772.768 |
| **E1** | La base de D1 más 12 LoRA de rango 32, uno por dominio, en QKV, salida de atención y las tres proyecciones del MLP. Enrutado duro por los `import` | 26.761.696 | 15.880.672 |
| **M1** | Switch top-1: los mismos 12 LoRA, con un router aprendido por bloque sobre un promedio causal del estado oculto. La salida se pondera por la probabilidad del gate y hay pérdida auxiliar de balance | 26.779.408 | 15.898.384 |

Entrenamiento:

- D1, D2 y M1: dos épocas (106,8M tokens).
- E1 parte del checkpoint final de D1: con la base congelada, cada adaptador se entrena una época solo con su dominio, 53,4M tokens en total.
- D1+ recibe esos mismos tokens extra, pero con datos de todos los dominios. Es el control que separa "aporta el adaptador" de "aporta entrenar más".

Comunes a todos: batch efectivo de 16.384 tokens, AdamW, LR 1e-3 con calentamiento de 500 pasos y coseno hasta el 10%, bf16. Cada archivo se lee como `<bos>` + tokens. La evaluación usa ventanas de 512 con paso 256.

## Errores corregidos antes de los números válidos

La revisión encontró varios fallos que habrían invalidado la comparación:

| Fallo | Efecto | Corrección |
| --- | --- | --- |
| El "router" de M1 comparaba cada embedding con los embeddings de los tokens 0–12 y no ponderaba por el gate | Enrutado fijo por identidad de token que no aprendía nada | Router `Linear` propio por bloque, salida ponderada por el gate y pérdida Switch |
| M1 calculaba los expertos con un bucle Python sobre máscaras | 5.172 tokens/s, el 70% del tiempo total | Expertos apilados `[E, d, r]` y tokens agrupados por experto con `bmm` |
| La base de E1 se entrenaba la mitad que D1 | Comparación injusta | E1 parte de D1, y se añadió el control D1+ |
| Los embeddings empezaban en N(0, 1) con salida atada | Logits enormes al inicio: D1 quedaba en PPC 4,155, peor que el n-grama | Inicialización N(0, 0,02) estilo GPT-2, salidas residuales escaladas por profundidad, LR 1e-3 |
| La evaluación anteponía 256 `<bos>` nunca vistos al entrenar | 10,1 nats/token en el inicio de cada archivo | Un solo `<bos>`, también al entrenar |
| El BPE byte-level parte algunos caracteres en dos tokens | Se contaban 795.825 caracteres en vez de 795.700 | Cada carácter cuenta una sola vez |
| Un corte de luz corrompió `M1.pt` mientras se escribía | Hubo que reentrenar M1 | Guardado atómico: archivo temporal y luego renombrado |

## Resultados

Una sola semilla, mismos 795.700 caracteres. Las columnas por tipo de texto están en nats por carácter:

| Modelo | PPC | Código | Comentarios | Strings | Formato |
| --- | ---: | ---: | ---: | ---: | ---: |
| D1 | 1,728 | 0,531 | 0,756 | 0,790 | 0,219 |
| **D1+** | **1,718** | 0,523 | 0,755 | 0,783 | 0,217 |
| D2 | 1,725 | 0,529 | 0,749 | 0,789 | 0,220 |
| E1, adaptador correcto | 1,724 | 0,528 | 0,754 | 0,787 | 0,217 |
| E1, solo base | 1,728 | 0,531 | 0,756 | 0,790 | 0,219 |
| E1, adaptador equivocado | 1,726 | 0,530 | 0,755 | 0,788 | 0,218 |
| M1 | 1,725 | 0,526 | 0,758 | 0,790 | 0,218 |
| N-grama BPE (ronda 2) | 3,279 | 1,347 | 1,137 | 1,494 | 0,584 |
| Sector-LM A (ronda 4) | 3,603 | 1,436 | 1,338 | 1,718 | 0,481 |
| `tiny_starcoder_py` | 1,296 | | | | |

Tiempos de entrenamiento aproximados en la RTX 3070, sacados de las horas de los checkpoints: D1 unos 26 min, D1+ 13, E1 14, D2 32 y M1 58. Los modelos retomados tras el corte de luz figuran con unos segundos en `results/round5/training_stats.json`.

## Lo que se descubrió

- **Los modelos neuronales funcionan.** Un transformer de 15M entrenado media hora mejora al mejor n-grama por casi un factor de 2.
- **Los cinco quedan a menos del 1% entre sí.** La diferencia probablemente cabe en la variación entre semillas.
- **Los adaptadores casi no aprendieron nada propio de su dominio.** E1 mejora a su base en un 0,2%, y con el adaptador equivocado da casi lo mismo que con el correcto.
- **Seguir entrenando gana a especializar**: D1+ supera a E1 con los mismos tokens extra.
- **Duplicar parámetros no sirve aquí.** Ni D2 ni M1 superan a D1+: con 53M tokens el límite son los datos, no la capacidad.

## Conclusión

A esta escala, especializar por dominio de Python **no mejora la calidad, pero tampoco la empeora**: E1 empata con D2 y con M1. Lo que E1 conserva es la ventaja de memoria, porque cambiar de dominio supone cargar un adaptador de unos 2 MB en bf16 mientras la base queda fija. El coste en velocidad está en [06_velocidad_y_memoria.md](06_velocidad_y_memoria.md).
