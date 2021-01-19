# NLP

The entire project is in **PORTUGUESE - BR**

This repository is a bunker where I register all my projects based on NLP models as well as models that can be used in NLP tasks (RNN, Transformer, etc).
All the techniques annotations that serve to explain step by step of the project are inside each project.

For simplicity, this repository can be useful to **study** and to **understand** the NLP models and tasks but it is not intended to be a NLP model repository.

The files inside Jupyter: The Dataset_for_Pretaining_Word_Embedding, Finding_Synonyms_and_Analogies and Pretraining Bidirectional_Encoder_Representations_from_Transformer(BERT) were made on Colaboratory, so the explanations are insider them

## Sentimental Analysis Dataset
Esse arquivo tem como obejtivo preparar e carregar dados. Os dados servirão para analizar emoções do autor de um determinado texto a partir de técnicas de classificação de sentimento em texto. Esse tipo de tarefa é denominado sentiment analysis.

Nesse projeto usaremos Stanford's Large Movie Review Dataset como o dataset da analise de sentimento. O texto contém :

* 25000 review de filme da IMDb para treino e para teste.
* Rotulos de 'positivo' ou 'negativo' para cada review na mesma proporção.

## Sentiment Analysis: Using RNN
Nessa seção, para tarefas de classificação de texto, nos aplicamos o modelo pré-treinado para vetorização de palavra GloVe e uma rede RNN bidirecional com multiplas camadas escondidas (hidden layers)

O modelo ira classificar se uma sequência de texto de tamanho indefinido contém emoções positiva ou negativas.

No modelo atual, cada palavra é vetorizada através da layer embedding. Então, codificamos (encode) cada vetorização de sequencia (união de palavras vetorizadas) usando um RNN bidirecional para obter a informação da sequencia desejada. Por ultimo, nos podemos concatenar os estados ocultos do bidirecional long-short term memory no inicio do time step e no final da time step e passar para a saida da camada de classificação (Linear) como informação de sequência de característica codificada.

No modelo criado BiRNN class temos :

* Embedding como a camada de embedding;
* LSTM como a camada escondida para codificação de sequencia;
* Uma camada Dense como camada de saida geradora de classificação de texto.

The projects are inspired at : 
https://d2l.ai/
