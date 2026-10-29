# Ronda 2: precio justo y sectores por dominio

## Hipótesis

1. Si Sector-LM paga por todo lo que pagan los demás, sabremos cuánto vale de verdad.
2. Agrupar los sectores por **dominio del archivo** en vez de por tipo de token debería reducir mucho las cargas de la caché sin perder calidad.

## Qué se cambió

Cuatro correcciones de la métrica, que se pueden activar por separado (`--price-sector`, `--format`, `--spelling`):

- **precio del sector**: se suma `log P(sector | sector anterior)` del HMM a cada predicción;
- **tokens de formato**: espacios, saltos de línea e indentación se predicen como tokens;
- **ortografía**: un token nunca visto paga carácter a carácter con un modelo de caracteres;
- **mismos caracteres**: la evaluación comprueba que se cubren exactamente los 795.700 caracteres del test.

Y un cambio de arquitectura: **sectores por dominio** (`--sectors domain`). Los `import` de cada archivo lo asignan a uno de 13 dominios, y todo el código de ese archivo va al sector de su dominio.

## Resultados

Ablación con sectores sintácticos (`results/ablation_*.json`; cada corrección por separado y luego todas juntas):

| Variante | PPC | Cargas por 1.000 tokens |
| --- | ---: | ---: |
| Sin correcciones | 2,088 | 209,5 |
| Solo precio del sector | 2,499 | 209,5 |
| Solo tokens de formato | 2,470 | 260,5 |
| Solo ortografía de desconocidos | 4,028 | 209,5 |
| **Todas las correcciones** | **6,141** | **260,5** |
| Todas, con sectores por dominio | **4,698** | **0,40** |
| N-grama BPE monolítico | 3,279 | |
| `tiny_starcoder_py` | 1,296 | |

Con sectores sintácticos, Viterbi acertó el 91,9% de los sectores.

## Lo que se descubrió

- El 2,088 inicial era **optimista por casi un factor de 3**. La mayor parte del engaño venía de los tokens nunca vistos, que salían gratis.
- Los sectores por dominio redujeron las cargas de **260 a 0,4 por cada 1.000 tokens**, unas 650 veces menos, y además **mejoraron** el PPC de 6,141 a 4,698. Partir por tipo de token cortaba el contexto de cada n-grama; partir por dominio lo conserva.
- Aun así, Sector-LM seguía peor que el n-grama BPE (4,698 frente a 3,279).

Diagnóstico para la ronda siguiente:

- el `<unk>` tenía probabilidad casi nula, alrededor de 1e-9, porque nunca se entrenaba;
- los comentarios y los strings son el 33% de los caracteres del test y seguían siendo un solo token cada uno.

## Conclusión

La métrica ya era justa. La idea de enrutar por dominio demostró su valor en **memoria**: casi nunca hay que cargar un sector. En **calidad** todavía no alcanzaba a la comparación justa.
