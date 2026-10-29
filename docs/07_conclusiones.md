# Conclusiones y siguientes pasos

## Qué quedó demostrado

1. **Enrutar por dominio ahorra memoria de verdad.** Con n-gramas, las cargas de sectores bajaron de 260 a 0,3 por cada 1.000 tokens. Con redes neuronales, cambiar de dominio supone cargar un adaptador de unos 4 MB (fp32) sobre una base fija.
2. **A esta escala, especializar por dominio de Python no da calidad.** E1, D2 y M1 empatan con D1 a menos del 1%, y seguir entrenando el modelo denso (D1+) gana a todos.
3. **El límite son los datos, no los parámetros.** Con 53M tokens, duplicar parámetros, como hacen D2 y M1, no mejora nada.
4. **El MoE token a token es el más caro.** M1 iguala a los demás en calidad, pero es 2,7 veces más lento al inferir y necesita todos sus expertos en memoria.
5. **Un transformer pequeño y bien entrenado gana a cualquier n-grama.** PPC 1,718 frente a 3,279, con 15M de parámetros y menos de una hora de una GPU de consumo.
6. **`tiny_starcoder_py` sigue muy por delante** (1,296), con 11 veces más parámetros y unas 600 veces más datos. Según su ficha en Hugging Face se entrenó con 32 A100 durante 18 horas sobre 100.000M tokens: unas 7.000 veces el cómputo de D1+ (6 × parámetros × tokens).

## Lo que no quedó demostrado

- **No hay varias semillas.** Las diferencias de menos del 1% de la ronda 5 podrían cambiar de orden con otra semilla.
- **El PPC no mide utilidad.** Un buen PPC no garantiza que el modelo escriba código correcto. Lo que genera `neural/generate.py` parece Python, pero no tiene sentido.
- **Jugamos en casa.** El test sale de la misma fuente que el train (codeparrot). En código de otro estilo la diferencia con StarCoder probablemente sería mayor.
- **Posible fuga a favor de StarCoder.** Sus datos y codeparrot salen de GitHub, así que pudo haber visto algunos archivos del test.

## Lecciones de método

- Antes de comparar, comprobar que todos los modelos pagan por los mismos caracteres. El primer PPC de 2,088 era optimista por casi un factor de 3.
- Comprobar que el entrenamiento y la evaluación ven el mismo tipo de contexto: ese error costó un 14% en la ronda 3.
- Cuando un resultado contradice lo esperado (un transformer peor que un n-grama), el fallo suele estar en el montaje y no en la idea: aquí fueron la inicialización y el `<bos>`.
- Incluir siempre un control que separe "aporta la idea" de "aporta más cómputo", como D1+.

## Siguientes pasos posibles

- **Especialistas por tarea en vez de por dominio.** Los dominios de Python se parecen demasiado entre sí, mientras que tareas como clasificar un fallo de CI, escribir un test o detectar un bug local son de verdad distintas. Ahí un adaptador por tarea sí podría compensar.
- **Detectores de bugs locales con datos sintéticos.** Se inyectan bugs en código correcto (argumentos intercambiados, `await` olvidado, `<` por `<=`) y se ajusta D1 como clasificador. Se mediría con precisión y recall sobre bugs reales, por ejemplo de BugsInPy.
- **Fusionar el LoRA con la base** al cargar un dominio, para que E1 corra exactamente a la velocidad de D1.
- **Más datos por dominio**, para saber si la especialización empieza a ganar cuando el modelo deja de estar limitado por los datos.
- **Varias semillas** en la ronda 5, para saber cuánto de esas diferencias es ruido.
