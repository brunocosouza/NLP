from d2l import torch as d2l
import torch
from torch import nn

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)

#Modelo RNN Bidirecional

class BiRNN (nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                num_layers, **kwargs):
        super(BiRNN,self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # bidirectional = True para um rede RNN birecional
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers = num_layers,
                              bidirectional = True)
        self.decoder = nn.Linear(4*num_hiddens, 2)
        
    def forward(self, inputs):
        # o formato da entrada (inputs) é (batch_size, num_palavras)
        # O LSTM precisa de uma sequencia como a primeira dimensão
        # A entrada é transformada em embedding e entao a caracteristica da palavra é extraida.
        # A saida tem formato (num_palavras, batch_size, dim_vetorização)
        embedding = self.embedding(inputs.T)
        # Como a entrada (embedding) é apenas o argumento para a camada LSTM, tanto h_0 como c_0 é zero inicialmente 
        # (verificar Modelo LSTM para compreender)
        # Apenas usamos a hidden states da ultima camada oculta em diferentes time step (output)
        # O formato da saida é (num_palavras, batch_size, 2 * num_hidden_units)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(embedding)
        # Concatenamos a hidden states da etapa inicial e da etapa final
        # Usamos como entrada para a camada Dense
        # O formato é, portanto, (batch_size, 4 * num_hiddens_units)
        encoding = torch.cat((outputs[0], outputs[-1]), dim = 1)
        outs = self.decoder(encoding)
        return outs
    
embed_size, num_hiddens, num_layer, devices = 100, 100 , 2, d2l.try_all_gpus()
model = BiRNN(len(vocab), embed_size, num_hiddens, num_layer)

# inicialização dos pesos
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])

model.apply(init_weights)

#Classe para carregar e instanciar o modelo glove.6b.100d
import os
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

# As funções a seguir são utilizada para treinar e mostrar o grafico de treinamento
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
    
#Parametros selecionados e chamada para o treinamento
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(model.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(model, train_iter, test_iter, loss, trainer, num_epochs, devices)


#Teste para verificação do modelo treinado
def predict_sentiment(net, vocab, sentence):
    sentence = torch.tensor(vocab[sentence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sentence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'

predict_sentiment(model, vocab, 'this movie is so great')
predict_sentiment(model, vocab, 'I hated this movie for me this was the worst movie ever made')
