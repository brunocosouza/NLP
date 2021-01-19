# coding: utf-8


from d2l import torch as d2l
import torch
from torch import nn
import os


batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)



# Convolução para uma entrada e um Kernel
def corr1D(X,K):
    W = K.shape[0]
    Y = torch.zeros((X.shape[0] - W + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + W]* K).sum()
    return Y

# Verificação do funcionamento
X = torch.tensor([0,1,2,3,4,5,6])
K = torch.tensor([1,2])
print(corr1D(X,K))




# Convolução para varios canais de entrada.
# Nesse caso necessitamos de varios kernel
# Porém, assim, como para um canal de entrada
# o output é singular (multiplicação seguida de soma)

def corr1d_multi_in(X,K):
    # Como o valor para cada posição da saida é a soma da multiplicação 
    # entre kernel e dados, é mais simples realizar uma operação em toda
    # a dimensão de um dado, e posteriormente, somar os resultados.
    return sum(corr1D(x,k) for x, k in zip(X,K))

X = torch.tensor([[0, 1, 2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5, 6, 7],
              [2, 3, 4, 5, 6, 7, 8]])
K = torch.tensor([[1, 2], [3, 4], [-1, -3]])
print(corr1d_multi_in(X, K))


# In[5]:


print([corr1D(x,k) for x, k in zip(X,K)])




# Para o modelo TextCNN vamos usar 2 camadas de embedding
# A primeira camada é de peso fixo e a outra para treinamento
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Essa camad de embedding não participa do treinamento
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        # Criação de multiplas camadas convolucionais unidimensional
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # Inicialmente precisamos concatenas o output das
        # duas camadas de embedding
        # (batch_size, num_words, word_vector_dim)
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # Conv1D requer a dimensão de canais de entrada como na dimensão das linhas
        # então permutaremos os dados embedding
        # (batch_size, embed_size, word_size)
        embeddings = embeddings.permute(0, 2, 1)
        # Embedding vai passar pelas camadas de convolução (de acordo com a quantidade escolhida)
        # O resultado ira para a camada de pooling que sobre a quantidade de canais
        # quantidade de canais de output de convs : (embed_size - kernel_size + 1)
        # O resultante da camada de pooling é um tensor : (batch_size, out_channel_size,1)
        # Removemos 1 com torch.squeeze e concatemos os valores de cada conv.
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        # Usamos a camada linear para classificação
        outputs = self.decoder(self.dropout(encoding))
        return outputs




embed_size, kernel_sizes, num_channels = 100, [3,4,5], [100,100,100]
devices = d2l.try_all_gpus()
model = TextCNN(len(vocab), embed_size, kernel_sizes, num_channels)

def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        
model.apply(init_weights)




class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r', encoding="utf8") as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)



# Corregar pre-trained word vector (Glove)
# O modelo vai vetorizar a palavra em um dominio de dim = 100
glove_embedding = TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
print(embeds.shape)


# Como nosso dataset de treinamento não é muito grande
# Vamos utilizar o modelo pré-treinado glove para fine-tuning a camada embedding
# do nosso modelo (model)
# Isso significa que nossa camada embedding vai saber vetorizar as palavras
# como um modelo treinado com uma grande quantidade de corpus.
model.embedding.weight.data.copy_(embeds)
model.embedding.weight.requires_grad = False


def train_batch_ch13(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        # Required for BERT Fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Store training_loss, training_accuracy, num_examples, num_features
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')



# Treinando o Modelo
lr, num_epochs = 0.001, 5
trainer = torch.optim.Adam(model.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
train_ch13(model, train_iter, test_iter, loss, trainer, num_epochs, devices)


def predict_sentiment(net, vocab, sentence):
    sentence = torch.tensor(vocab[sentence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sentence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'

print(predict_sentiment(model, vocab, 'this movie is so great'))
print(predict_sentiment(model, vocab, 'I hated this movie for me this was the worst movie ever made'))

