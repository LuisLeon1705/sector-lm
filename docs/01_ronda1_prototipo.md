# Ronda 1: el prototipo

## Hipótesis

Un modelo de n-gramas dividido en sectores sintácticos, con tokens estructurados, Viterbi para los tokens ambiguos y una tabla de símbolos para copiar nombres, puede competir con un n-grama BPE monolítico y ocupar menos memoria activa, porque solo carga los sectores que usa.

## Qué se construyó

La primera versión se escribió con Antigravity y la terminó Codex:

- parser de Python con tree-sitter que etiqueta cada token con su sector, regla y condición;
- un trigrama Kneser-Ney por sector;
- HMM con Viterbi para asignar sectores;
- tabla de símbolos con copia de nombres definidos;
- almacenamiento por sector en disco con `memmap2` y caché LRU;
- `sector_eval`: el binario que entrena, evalúa y compara con el n-grama BPE y con `tiny_starcoder_py`.

## Problemas encontrados en la revisión

La primera ejecución completa no terminaba: después de 40 minutos seguía corriendo.

| Problema | Efecto | Corrección |
| --- | --- | --- |
| El top-k recorría todo el vocabulario en cada predicción | La evaluación tardaba horas | Top-k exacto con parada temprana: algoritmo de umbral con cotas superiores por orden de n-grama |
| `get_kth_score` ordenaba la lista completa en cada iteración | Coste cuadrático escondido | Montículo de tamaño k |
| La variante "sin copia" del top-k doble reutilizaba la lista con copia | Resultados no exactos | Dos listas independientes, con un test de equivalencia frente a la búsqueda exhaustiva |
| Un solo archivo de test enorme dominaba el tiempo | Parecía colgado | Evaluación paralela con rayon y registro de los archivos más lentos |
| La barra de progreso no aparecía si la salida no era una terminal | Imposible saber si avanzaba | Barra que imprime también a un log |

## La métrica todavía no era justa

Con estos arreglos el programa terminaba, pero el PPC salía demasiado bueno porque el modelo no pagaba por varias cosas que sí pagan los demás:

- **cada comentario o string era un solo token**: costaba casi lo mismo uno de 5 caracteres que uno de 200;
- **el sector de cada token se daba gratis**: no se cobraba la probabilidad de cambiar de sector;
- **los espacios, saltos de línea e indentación no se predecían**;
- **los tokens nunca vistos no costaban su ortografía**.

La ronda 2 midió cuánto influía cada una de estas ventajas y las corrigió todas.

## Conclusión

La arquitectura funcionaba de punta a punta, pero ni la velocidad ni la métrica permitían todavía sacar conclusiones. La lección: **antes de comparar modelos hay que asegurarse de que todos pagan por los mismos caracteres**.
