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

## Sentiment Analysis: Using CNN

No modelo de linguagem do projeto *Sentiment Analysis: Using RNN* tratamos dados de texto como serie temporais e, portanto, com apenas uma dimensão e usamos RNN para processar os dados. 

Como contrapartida, em modelos para identificação de imagem, como Redes Neurais Convolucionais (CNN), normalmente, tratamos a imagem como dados de duas dimensões e, assim, exploramos as técnicas (convolução) de analise de imagem para duas dimensões Entretando, nesse projeto, usaremos redes convolucionais para apenas uma dimensão. Dessa maneira podemos capturar associações entre palavras adjacentes. 

O passo a passo do projeto é dado a seguir: 


1.   Usaremos modelo  pré-treinado GloVe para vetorizar as palavras;
2.   Usaremos a arquitetura CNN para treinamento;
3.   Com o modelo treinado classificaremos o texto de acordo com a analise de sentimento.

### Teoria Camada de Convolução unidimensional
Camadas de convolução de uma dimensão funcionam da seguinte maneira : esse tipo de camada realiza uma operação denominada *cross-correlation*. Nessa operação temos os dados em uma dimensão, que serão convolucionados, e, um kernel (uma janela de valores numéricos que realiza a operação de cross-correlation). A operação se inicia da esquerda para a direita. O kernel multiplica seus valores com os primeiros valores dos dados e então soma essa multiplicação, resultando em um valor unico final. Em seguida, o kernel pula uma posição posterior e realiza a mesma operação.

Para melhor entendimento valos considerar:

1. Dados : (0, 1, 2, 3, 4, 5, 6)
2. Kernel : (1, 2)

Portanto, o output da correlação : (2, 5, 8, 11, 14, 17).

Vale ressaltar que o tamanho do output é dado como : **(data_size - kernel_size +1)**

Na pratica, as operações de cross-correlation pode ser realizada para multiplos canais de entrada (quantidade de dados) com diversos kernels,resultando em apenas um canal de output ou varios canais de output (dependendo da aplicação do modelo). 

O output resultante de uma operação com diversos canais de saida, passa por uma camada de pooling. Essa camada de pooling realiza o subdimensionamento dos dados resultantes.

Para mais detalhes, Convolution Neural Network (Capitulo 40 - 47) : http://deeplearningbook.com.br/capitulos/

### Modelo TextCNN

O modelo treinado nesse projeto é o TextCNN model.
Esse modelo utiliza uma camada de convolução unidimensional, camada de max-over-time pooling e uma camada Linear.

* A camada de convolução é definido por multiplos kernels. O output resultante apresenta dimensão correspondente (discutido anteriormente) para multiplos canais, ou seja, é uma saida de multiplos canais. 
* Os canais de output passam pela camada de pooling sobre o tempo (dados sequenciais). Os varios valores resultantes da camada de pooling sobre a sequencia de palavras é então contatenado e seu tamanho é correspondente à quantidade de canais de saida.
* A resultante é transformada em vetor e serve de entrada para uma camada linear de classificação.
