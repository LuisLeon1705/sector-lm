# Ronda 4: la misma historia al entrenar y al evaluar

## Hipótesis

Si cada modelo de sector usa la misma historia al entrenar y al evaluar, desaparece el empeoramiento de la ronda 3. Quedaba por saber qué historia es mejor: la del propio sector o la secuencia global.

## Qué se cambió

`--history` con dos variantes (`src/model/history.rs`):

- **A, `sector`** (por defecto): la historia son los dos tokens anteriores **del mismo sector**, igual al entrenar y al evaluar.
- **B, `mixed`**: la historia son los dos tokens anteriores de la secuencia global. Los de otros sectores se representan con un marcador `<sec:NN>`, para que el contexto diga "aquí hubo un comentario" sin convertirlo en `<unk>`.

El test `training_and_evaluation_contexts_match_for_each_target` comprueba que, para cada objetivo, el contexto que consulta la evaluación tiene la misma forma que los contextos contados al entrenar.

## Resultados

Sobre los mismos 100 archivos y 795.700 caracteres (`results/round4/`):

| Variante | PPC |
| --- | ---: |
| Ronda 3 (historias desajustadas) | 4,183 |
| **A: historia del mismo sector** | **3,603** |
| B: historia global con `<sec:NN>` | 3,648 |
| N-grama BPE monolítico | 3,279 |
| `tiny_starcoder_py` | 1,296 |

Nats por carácter por tipo de texto:

| Tipo | A | B | BPE |
| --- | ---: | ---: | ---: |
| Código | 1,436 | 1,436 | 1,347 |
| Comentarios | 1,338 | 1,352 | 1,137 |
| Strings | 1,718 | 1,736 | 1,494 |
| Formato | 0,481 | 0,506 | 0,584 |

Las cargas se mantuvieron en 0,307 por cada 1.000 tokens.

## Por qué A gana a B

Justo después de un comentario, A salta el comentario y le da al modelo de código sus dos últimos tokens de código, que dicen mucho. B le da `<sec:TP>, <sec:TP>`, que solo dice "acabó un comentario". En el código empatan, porque casi siempre viene detrás de más código. B pierde un poco en el resto.

## Conclusión

Corregido el desajuste, Sector-LM mejoró un 14%. Pero **sigue perdiendo contra el n-grama BPE en todo menos en formato**. Con n-gramas, partir en sectores no da más calidad que un modelo único; lo único que aporta es la carga perezosa.

Ese es el límite de los n-gramas, que no pueden aprovechar más que dos tokens de contexto. Por eso la ronda 5 pasa a modelos neuronales.
