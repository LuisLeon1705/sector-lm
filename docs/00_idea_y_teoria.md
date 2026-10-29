# La idea y la teoría

## El punto de partida

Un LLM normal es un solo bloque: todos sus parámetros participan en cada token, y para usarlo hay que tener el modelo entero en memoria. La pregunta de Sector-LM fue si un modelo de lenguaje podía funcionar "en paralaje": dividido en **sectores** especializados que se cargan solo cuando hacen falta, en lugar de una única cadena de cómputo.

El dominio elegido fue **código Python**, porque tiene estructura explícita (el árbol sintáctico) que permite decidir con reglas a qué sector pertenece cada token.

## Tokens estructurados `SS R C IIIIII`

Cada token se representa con un identificador compuesto:

| Campo | Significado |
| --- | --- |
| `SS` | Sector: el submodelo responsable del token |
| `R` | Regla sintáctica: definición, referencia, palabra clave, literal... |
| `C` | Condición o contexto: por ejemplo, si el token está en un comentario o es formato |
| `IIIIII` | Identificador del token dentro de su sector |

Así, el mismo texto puede vivir en sectores distintos según su papel, y cada sector tiene un vocabulario pequeño.

## Dos problemas que había que resolver

1. **¿A qué sector pertenece un token ambiguo?** Por ejemplo, `def` como palabra clave frente a `def` dentro de un string. Se usó un **HMM con Viterbi** sobre la secuencia: los estados son sectores y el modelo elige la secuencia de sectores más probable. Con las etiquetas del parser, Viterbi acertó el 91,9% de los sectores en la ronda 2.
2. **¿Cómo predice un sector un nombre definido en otro?** Si `Animal` se define con `class Animal`, más adelante el sector de uso tiene que poder "copiar" ese nombre. Para eso hay una **tabla de símbolos**: las definiciones se registran y las referencias (regla `R`) pueden copiarse de ella con un peso aprendido.

## Los submodelos

En las rondas 1 a 4 cada sector es un **n-grama de Kneser-Ney** (trigrama) con:

- top-k exacto con parada temprana, para no recorrer todo el vocabulario;
- almacenamiento por sector en disco con `memmap2` y una **caché LRU** de sectores cargados;
- un modelo de **ortografía por caracteres** para tokens nunca vistos.

En la ronda 5 los submodelos pasan a ser **transformers pequeños** (ver [05_ronda5_expertos_neuronales.md](05_ronda5_expertos_neuronales.md)).

## Sectores sintácticos frente a sectores por dominio

- **Sintácticos** (ronda 1): un sector por tipo de token (nombres, palabras clave, operadores, literales...). Un archivo usa casi todos los sectores todo el tiempo, así que la caché carga y descarga sin parar: unas 260 cargas por cada 1.000 tokens.
- **Por dominio** (ronda 2 en adelante): los `import` de cada archivo lo asignan a uno de 13 dominios (científico, web, testing, sistema...). Un archivo entero usa un solo sector de código más el de texto, y las cargas bajan a unas 0,3 por cada 1.000 tokens.

## Cómo se mide

- **PPC (perplejidad por carácter)** = `exp(nats totales / caracteres)`. Mide cuánto le sorprende al modelo el texto real. Menos es mejor. Se mide por carácter y no por token porque cada modelo corta el texto de forma distinta: por carácter, todos compiten sobre lo mismo.
- **Conjunto de test fijo**: los primeros 100 archivos de `data/test.jsonl`, exactamente **795.700 caracteres**. Todos los modelos de todas las rondas se evalúan sobre esos mismos caracteres.
- **Nats por carácter por tipo de texto**: el total separado en código, comentarios, strings y formato (espacios, saltos de línea, indentación). Indica dónde gana o pierde cada modelo.
- **Cargas por cada 1.000 tokens**: cuántas veces la caché LRU tiene que cargar un sector desde disco. Es la medida del ahorro de memoria.

## Modelos de comparación

- **N-grama BPE monolítico**: un único trigrama Kneser-Ney sobre tokens BPE de 16.000 entradas. Es la comparación justa para las rondas de n-gramas: misma familia, mismos datos.
- **`bigcode/tiny_starcoder_py`** (164M de parámetros): un transformer real de código Python. No es un rival a su escala sino una referencia de techo: dice cuánto margen queda.

## Trabajos relacionados

La idea de repartir un modelo en expertos y activar solo una parte tiene mucha historia. La mencionamos solo por nombre, como contexto:

- Mixture of Experts y Switch Transformer: enrutado aprendido token a token.
- Mixtral: un MoE abierto de gran tamaño.
- DEMix y Branch-Train-Merge (BTM): expertos por dominio entrenados por separado.
- LoRA y S-LoRA: adaptadores de bajo rango y cómo servir muchos sobre una base compartida.
- Offloading de expertos de MoE: cargar en la GPU solo los expertos que se usan.

La aportación de este proyecto no es la idea en sí, sino **medirla con cuidado a escala pequeña**, en una sola GPU de consumo, frente a comparaciones justas.
