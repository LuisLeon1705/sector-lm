# Velocidad y memoria

Medido en una RTX 3070 (8 GB) con pesos en fp32 y cálculo en bf16, un modelo a la vez y con la GPU limpia. Lote de 8 secuencias de 512 tokens, con entradas aleatorias, porque la velocidad no depende del contenido. Scripts:

- [`scripts/bench_round5_speed.py`](../scripts/bench_round5_speed.py): D1, D1+, D2, E1 y M1, a partir de los checkpoints de la ronda 5;
- [`scripts/bench_starcoder_speed.py`](../scripts/bench_starcoder_speed.py): `tiny_starcoder_py`, desde la caché local de Hugging Face.

Los tokenizadores dan 3,67 caracteres por token (el BPE de la ronda 5) y 3,73 (StarCoder), así que la velocidad se da también en caracteres por segundo, que es lo comparable.

## Resultados

| Modelo | PPC | Inferencia en lote | Latencia, 1 secuencia | Entrenamiento | Pesos necesarios (fp32) | Pico al inferir | Pico al entrenar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D1 / D1+ | 1,718 | 257.000 tok/s (944.000 car/s) | 3,5 ms | 87.000 tok/s | 60 MB | 287 MB | 1,6 GB |
| D2 | 1,725 | 206.000 tok/s | 3,3 ms | 67.000 tok/s | 107 MB | 381 MB | 1,9 GB |
| E1 | 1,724 | 210.000 tok/s (771.000 car/s) | 7,0 ms | 83.000 tok/s (solo adaptadores) | 64 MB (base + 1 adaptador) | 350 MB* | 1,6 GB |
| M1 | 1,725 | 94.000 tok/s (346.000 car/s) | 15,5 ms | 33.000 tok/s | 107 MB (los 12 expertos) | 490 MB | 3,1 GB |
| `tiny_starcoder_py` | 1,296 | 32.700 tok/s (122.000 car/s) | 21,0 ms | 7.200 tok/s, solo con lote 2 | 690 MB | 1.417 MB | 4,7 GB (lote 2) |

\* Medido con los 12 adaptadores cargados. En uso real solo hace falta uno: 4 MB más que D1.

Con el lote de 8 de nuestros modelos, StarCoder **no cabe en 8 GB al entrenar**: pide unos 12 GB, Windows lo pasa a la RAM compartida y baja a unos 900 tokens/s.

La cifra de 7.824 tokens/s de `results/starcoder_metrics.json` es otra medición: lote 1, ventanas de 2.048, fp16 y la evaluación completa. No es comparable con esta tabla.

## Lectura

- **M1 sale caro.** Para la misma calidad, es 2,7 veces más lento al inferir, 4,4 veces más lento en latencia y usa el doble de memoria al entrenar. Además necesita los 12 expertos siempre en memoria, porque cambia de experto token a token. Repartir los tokens entre expertos tiene un coste fijo grande en modelos tan pequeños.
- **E1 cumple en memoria.** Con un dominio cargado ocupa lo que D1 más 4 MB, y cambiar de dominio supone cargar esos 4 MB.
- **El 18% de velocidad que pierde E1 tiene arreglo.** El LoRA se calcula aparte en cada capa, cinco multiplicaciones pequeñas más. Como un archivo usa siempre el mismo dominio, el adaptador **se puede fusionar con los pesos de la base** al cargarlo (`W + B·A`). Así E1 correría a la velocidad exacta de D1, y deshacer la fusión al cambiar de dominio cuesta milisegundos. No está implementado.
- **D2** gasta un 20% más de tiempo y casi el doble de pesos para ganar un 0,2%.
- **Frente a StarCoder**, D1+ es unas 8 veces más rápido al inferir, 11 veces más pequeño en pesos y unas 12 veces más rápido al entrenar. StarCoder necesita menos de la mitad de información por carácter (0,26 frente a 0,54 nats).

A igual calidad, el orden en eficiencia es **D1+ ≈ E1 fusionado > D2 > M1**.
