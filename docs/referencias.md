# Orígenes de las ideas

Las ideas que usa o con las que se compara Sector-LM, agrupadas por tema. Las marcadas con ✔ se comprobaron en la web al escribir este documento: título, autores, año e identificador. Las demás se citan de memoria con su identificador habitual; conviene confirmarlas antes de citarlas formalmente.

## Modelos de lenguaje clásicos (rondas 1–4)

| Idea | Origen | Dónde se usa aquí |
| --- | --- | --- |
| Suavizado Kneser-Ney | Kneser y Ney, *Improved backing-off for M-gram language modeling*, ICASSP 1995. Comparativa de referencia: Chen y Goodman, *An Empirical Study of Smoothing Techniques for Language Modeling*, Harvard TR-10-98, 1998 | Los n-gramas de cada sector y la comparación BPE |
| Algoritmo de Viterbi y HMM | Viterbi, *Error bounds for convolutional codes and an asymptotically optimum decoding algorithm*, IEEE Trans. Information Theory, 1967. Tutorial clásico: Rabiner, *A Tutorial on Hidden Markov Models*, Proc. IEEE, 1989 | Asignar sector a los tokens ambiguos |
| BPE para modelos de lenguaje | Sennrich, Haddow y Birch, *Neural Machine Translation of Rare Words with Subword Units*, ACL 2016 (arXiv 1508.07909) | Tokenizador de la comparación, BPE de texto y tokenizador de la ronda 5 |
| Mecanismo de copia | Gu et al., *Incorporating Copying Mechanism in Sequence-to-Sequence Learning* (CopyNet), ACL 2016 | Idea cercana a la tabla de símbolos que copia nombres definidos |

## Mezcla de expertos y enrutado

| Idea | Origen | Relación con Sector-LM |
| --- | --- | --- |
| MoE con compuerta dispersa | Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, ICLR 2017 (arXiv 1701.06538) | La idea base de activar solo una parte del modelo |
| MoE a gran escala | Lepikhin et al., *GShard*, 2020 (arXiv 2006.16668) | Escalado de MoE entre muchas GPU |
| Switch Transformer (top-1) | Fedus, Zoph y Shazeer, *Switch Transformers*, JMLR 2022 (arXiv 2101.03961) | M1 sigue su diseño: top-1, salida ponderada por el gate, pérdida de balance |
| Mixtral | Jiang et al., *Mixtral of Experts*, 2024 (arXiv 2401.04088) | MoE abierto de referencia |
| ✔ Expertos por dominio | Gururangan, Lewis, Holtzman, Smith y Zettlemoyer, *DEMix Layers: Disentangling Domains for Modular Language Modeling*, NAACL 2022 (arXiv 2108.05036) | Lo más parecido a E1: expertos por dominio que se pueden añadir o quitar |
| ✔ Branch-Train-Merge (BTM) | Li, Gururangan, Dettmers, Lewis, Althoff, Smith y Zettlemoyer, *Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models*, 2022 (arXiv 2208.03306) | Expertos entrenados por separado desde una semilla compartida, como E1 desde D1 |
| ✔ c-BTM | Gururangan et al., *Scaling Expert Language Models with Unsupervised Domain Discovery*, 2023 (arXiv 2303.14177) | Descubre los dominios con k-means en vez de reglas. Encuentra que la ganancia depende de que los clusters sean significativos, lo que encaja con nuestro resultado |
| ✔ Branch-Train-MiX (BTX) | Sukhbaatar et al., *Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM*, COLM 2024 (arXiv 2403.07816) | Une expertos de dominio en un MoE con enrutado aprendido: el puente entre E1 y M1 |

## Enrutado entre modelos enteros

| Idea | Origen | Relación |
| --- | --- | --- |
| ✔ RouteLLM | Ong et al., *RouteLLM: Learning to Route LLMs with Preference Data*, ICLR 2025 (arXiv 2406.18665) | Un router elige entre un modelo fuerte y uno débil para ahorrar coste |
| FrugalGPT | Chen, Zaharia y Zou, *FrugalGPT*, 2023 (arXiv 2305.05176) | Cascadas de modelos de distinto coste |

## Adaptadores LoRA

| Idea | Origen | Relación |
| --- | --- | --- |
| LoRA | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022 (arXiv 2106.09685) | Los adaptadores de E1 y los expertos de M1 |
| S-LoRA | Sheng et al., *S-LoRA: Serving Thousands of Concurrent LoRA Adapters*, 2023 (arXiv 2311.03285) | Servir muchos adaptadores sobre una base compartida |
| ✔ LoraHub | Huang, Liu, Lin, Pang, Du y Lin, *LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition*, COLM 2024 (arXiv 2307.13269) | Combinar adaptadores de tareas distintas |

## Cargar solo lo necesario

| Idea | Origen | Relación |
| --- | --- | --- |
| ✔ Offloading de expertos | Eliseev y Mazur, *Fast Inference of Mixture-of-Experts Language Models with Offloading*, 2023 (arXiv 2312.17238) | Ejecutar Mixtral en hardware de consumo cargando solo los expertos en uso: la misma motivación que la caché de sectores |
| LLM in a Flash | Alizadeh et al., *LLM in a flash: Efficient Large Language Model Inference with Limited Memory*, 2023 (arXiv 2312.11514) | Leer pesos desde almacenamiento flash bajo demanda |

## Escalado y entrenamiento

| Idea | Origen | Relación |
| --- | --- | --- |
| Leyes de escalado | Kaplan et al., *Scaling Laws for Neural Language Models*, 2020 (arXiv 2001.08361) | La pérdida baja como una potencia del cómputo: los primeros pasos son baratos |
| Chinchilla | Hoffmann et al., *Training Compute-Optimal Large Language Models*, 2022 (arXiv 2203.15556) | Unos 20 tokens por parámetro: con 53M tokens, D2 y M1 están limitados por los datos |
| The Bitter Lesson | Sutton, ensayo de 2019 | Los métodos generales que escalan con cómputo ganan a la estructura diseñada a mano, como pasó al pasar de n-gramas a transformers |
| Inicialización de GPT-2 | Radford et al., *Language Models are Unsupervised Multitask Learners*, 2019 | Pesos N(0, 0,02) y salidas residuales escaladas por profundidad: el arreglo de la ronda 5 |
| ✔ `tiny_starcoder_py` | Ficha de [`bigcode/tiny_starcoder_py`](https://huggingface.co/bigcode/tiny_starcoder_py): 164M de parámetros, unas 6 épocas sobre el Python de StarCoderData (100.000M tokens), 32 A100 durante 18 horas | El modelo de referencia |
| StarCoder | Li et al., *StarCoder: may the source be with you!*, 2023 (arXiv 2305.06161) | La familia de la referencia |

## Testing y análisis de código (ideas abiertas)

| Idea | Origen | Relación |
| --- | --- | --- |
| ✔ TestGen-LLM | Alshahwan et al., *Automated Unit Test Improvement using Large Language Models at Meta*, FSE 2024 (arXiv 2402.09171) | Generar tests y quedarse solo con los que compilan, pasan de forma estable y suben la cobertura |
| DeepBugs | Pradel y Sen, *DeepBugs: A Learning Approach to Name-based Bug Detection*, OOPSLA 2018 (arXiv 1805.11683) | Detectar bugs locales entrenando con bugs inyectados artificialmente |
| ✔ PrimeVul | Ding et al., *Vulnerability Detection with Code Language Models: How Far Are We?*, 2024 (arXiv 2403.18624) | Con datos limpios, un modelo de 7B pasa del 68% al 3% de F1: el ML solo detecta mal vulnerabilidades |
| ✔ RacerD | Blackshear, Gorogiannis, O'Hearn y Sergey, *RacerD: Compositional Static Race Detection*, OOPSLA 2018 (DOI 10.1145/3276514) | Detección estática de data races en Java |
| ThreadSanitizer | Serebryany e Iskhodzhanov, *ThreadSanitizer: data race detection in practice*, WBIA 2009 | Detección dinámica de data races |
| Tricorder | Sadowski et al., *Tricorder: Building a Program Analysis Ecosystem*, ICSE 2015 | Aprender de lo que hacen los desarrolladores con cada aviso de análisis estático |
| Defects4J / BugsInPy | Just, Jalali y Ernst, ISSTA 2014 (Java); Widyasari et al., FSE 2020 (Python) | Bancos de bugs reales para medir detectores |

## Fuentes consultadas para las marcas ✔

- [Ficha de bigcode/tiny_starcoder_py](https://huggingface.co/bigcode/tiny_starcoder_py)
- [DEMix, arXiv 2108.05036](https://arxiv.org/abs/2108.05036v1) · [Branch-Train-Merge, arXiv 2208.03306](https://arxiv.org/pdf/2208.03306) · [c-BTM, arXiv 2303.14177](https://arxiv.org/abs/2303.14177v1) · [BTX, arXiv 2403.07816](https://arxiv.org/abs/2403.07816v1)
- [RouteLLM, arXiv 2406.18665](https://arxiv.org/abs/2406.18665v4) · [LoraHub, arXiv 2307.13269](https://arxiv.org/abs/2307.13269v3) · [Offloading de MoE, arXiv 2312.17238](https://arxiv.org/abs/2312.17238)
- [TestGen-LLM, arXiv 2402.09171](https://arxiv.org/pdf/2402.09171) · [PrimeVul, arXiv 2403.18624](https://arxiv.org/abs/2403.18624v2) · [RacerD, SPLASH 2018](https://2018.splashcon.org/details/splash-2018-OOPSLA/1/RacerD-Compositional-Static-Race-Detection)
