from d2l import torch as d2l
import torch
from torch import nn
import os

# Primeiro fazemos o download do dataset para :
# "../data" e extraimos o "../data/aclimdb"

d2l.DATA_HUB['aclImdb'] = (
    'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
    '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')

def read_imdb(data_dir, is_train):
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test', label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

train_data = read_imdb(data_dir, is_train = True)
print('# training:', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('label: ', y, 'review: ', x[0:60])
    
# Tokenizar e vocabulario
# cada palavra sera um token e a partir disso criaremos um dicionario baseado no dataset de treinamento

train_tokens = d2l.tokenize(train_data[0], token = 'word')
vocab = d2l.Vocab(train_tokens, min_freq = 5, reserved_tokens = ['<pad>'])

# Com esse grafico, verificamos que a maioria das frases no train_tokens tem entre 100 e 200 tokens.
# ou seja, a maioria das review tem entre 100 e 200 palavras aproximadamente

d2l.set_figsize()
d2l.plt.hist([len(line) for line in train_tokens], bins = range(0,1000,50))

# Padding to the Same Length
# Portanto, como cada review tem tamanhos distintos, vamos equalizar o comprimento com padding.
# Fazemos isso para combinar depois os dados em minibatch para treinamento do modelo de analise de sentimento
# Vamos fixar o tamnho de cada review em 500 tokens diminuindo o tamanho ou adicionando <unk>.

def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))

num_steps = 500
train_feature = torch.tensor([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
print(train_feature.shape)

# Agora vamos criar um iterator de dados. Cada iteração vai retornar um minibatch dos dados

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

train_iter = d2l.load_array((train_feature, torch.tensor(train_data[1])), 64)

for X, y in train_iter:
    print('X:', X.shape, 'y: ', y.shape)
    break
    
print('# batches : ', len(train_iter))

# Colocando tudo junto

def load_data_imdb(batch_size, num_steps=500):
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
