# Ronda 3: `<unk>` aprendido y BPE para comentarios y strings

## Hipótesis

1. Si el `<unk>` se entrena con los tokens que aparecen una sola vez, el modelo dejará de pagar cifras absurdas por lo desconocido.
2. Partir el contenido de comentarios y strings en piezas BPE debería abaratar ese 33% de los caracteres.
3. Ponderar la ortografía por la frecuencia de cada token debería ajustarla mejor a lo que aparece en el texto.

## Qué se cambió

- `--unk-singletons`: los tokens vistos una sola vez en entrenamiento cuentan como `<unk>` en los n-gramas. Se conservan en el vocabulario de ortografía.
- `--text-bpe`: el contenido de comentarios y strings se tokeniza con un BPE propio de 8.000 entradas, en un sector compartido `TextPiece`. Las comillas, los prefijos y el `#` siguen siendo tokens de sintaxis.
- `--spelling-frequency`: el modelo de caracteres se entrena ponderando cada token por su frecuencia.

## Resultados

Ablación con sectores por dominio (`results/round3/`):

| Etapa | Cambio | PPC |
| --- | --- | ---: |
| Partida (ronda 2) | | 4,698 |
| `<unk>` por singletons | `00` → `01` | **3,961** |
| BPE de comentarios y strings | `01` → `02` | 4,082 |
| Ortografía por frecuencia | `02` → `03` | 4,183 |
| N-grama BPE monolítico | | 3,279 |
| `tiny_starcoder_py` | | 1,296 |

Nats por carácter del modelo final, por tipo de texto:

| Tipo | Caracteres | Sector-LM | BPE |
| --- | ---: | ---: | ---: |
| Código | 309.647 | 1,456 | 1,347 |
| Comentarios | 96.241 | 1,404 | 1,137 |
| Strings | 209.497 | 2,157 | 1,494 |
| Formato | 180.315 | 0,560 | 0,584 |

Las cargas bajaron a 0,307 por cada 1.000 tokens (unos 4 MB). Los tokens nunca vistos fueron el 3,88% de los predichos, y su ortografía costó 2,714 nats por carácter (106.503 caracteres).

## El error escondido

El `<unk>` aprendido funcionó: mejoró 0,74 puntos. Pero el BPE de texto **empeoró** el resultado, cuando tenía que mejorarlo. La revisión encontró la causa:

- **al entrenar**, cada modelo de sector veía solo los tokens de su propio sector, seguidos;
- **al evaluar**, la historia eran los dos tokens anteriores de la secuencia **mezclada**.

Mientras todo el archivo estaba en un solo sector, las dos cosas coincidían. Al sacar los trozos de comentario al sector `TextPiece`, el modelo de código empezó a ver al evaluar, en su historia, trozos de comentario que nunca había visto al entrenar, convertidos en `<unk>`. Al sector `TextPiece` le pasaba lo mismo al revés.

Una prueba rápida, que usaba al evaluar la historia del mismo sector, bajó el PPC de **4,183 a 3,603** sin tocar nada más.

## Conclusión

El `<unk>` aprendido era la mejora que faltaba. El BPE de texto era buena idea, pero destapó un desajuste entre entrenamiento y evaluación que existía desde el principio. Se corrigió en la ronda 4.
