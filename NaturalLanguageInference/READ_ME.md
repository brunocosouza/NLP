# Natural Language Inference and the Dataset

Inferencia de linguagem natural é a tarefa de identificar se uma *hipotese* pode ser relacionada à uma *premissa*. NLI pode decidir se uma sentença é inferida à outra ou eliminar redundancia identificando sentenças que são semanticalmente equivalentes.

Mais especificamente, NLI determina a relação logica em um par de texto. Relacões podem ser de 3 tipos:

* Entailment : a hipotese pode ser inferida à uma premissa
* Contradiction : a negação da hipotese pode ser inferida à uma premissa
* Neutral : todos os outros casos.

Por exemplo, o texto a seguir pode ser classificado como *entailment* porque uma hipotese pode ser considerada na premissa:

> Premissa : Duas pessoas estão se abraçando

> Hipotese : Duas pessoas estão mostrando afeto.

## Natural Language Inference: Using Attention e Stanford NLI Dataset

### Stanford NLI
O Dataset de Stanford para NLI é uma coleção de mais de 500.000 pares de sentenças rotuladas.

Nesse etapa do projeto, definimos uma função que retorna apenas parte do dataset com uma lista de premissas, hipoteses e os rotulos entre eles (devido a dimensão do dataset).

O dataset possui pares de frases correspondendo à *Entailment*, ou, *Contradiction*, ou, *Neutral*.

**OBS : Caso você esteja utilizando Windows, aconselho baixar manualmente o dataset e colocar dentro de uma pasta denominada "data" no mesmo local do projeto. O dataset "snli_1.0" deve ficar, portanto, no caminho "../data/snli_1.0"**.

### Modelo

Para o treinamento de técnicas de inference de linguagem natural utilizaremos um modelo de atenção denominado *decompasable attention model* (Parikh et al, 2016).

O Modelo resumidamente possui três etapas, Attend, Compare e Aggregate.

* A etapa Attend é responsavel por alinhar cada palavara da premisa com a palavra da hipotese (em uma matriz).

* A etapa Compare realiza o alinhamento entre par de palavra de forma 'hard way' (utilizando pesos para associações). 

* A etapa Aggregate realiza a soma do resultado provindo da camada de Compare e posteriormente a concatenação dos dados para fornece-los ao modelo de classificação.

Os attention weights calculados para alinhamento das palavras da premissa com as palavras da hipotese são calculados como: 

> $e_ij = f(a_i)^T f(b_j)$,

onde $f$ é um MLP.

Para a Compare realizar a comparação 'hard way', calculamos os valores *beta* e *alpha* :

> $beta_i = softmax(f_a) * b$

> $alpha_i = softmax(f_b) * a$

Então, alimentamos a concatenação de palavras de uma sequência e palavras alinhadas de outra sequência em uma função g (um MLP). O resultado são dois sets de dados $V_A$ e $ V_b$, dados de comparação entre as palavras na premissa com todas as palavras na hipotese e vice versa.

Apos a comparação hard-way, a ultima etapa é agrega toda a informação somando os conjuntos de output da etapa de Compare.
