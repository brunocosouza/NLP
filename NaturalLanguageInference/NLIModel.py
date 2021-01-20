#!/usr/bin/env python
# coding: utf-8

#Stanford Natural Language Inference Dataset (SNLI)

from d2l import torch as d2l
import torch
from torch import nn
import os
import re
from torch.nn import functional as F


#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')
d2l.DATA_HUB['SNLI']


def download(name, cache_dir=os.path.join('..', 'data')):
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in d2l.DATA_HUB, f"{name} does not exist in {d2l.DATA_HUB}."
    url, sha1_hash = d2l.DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = d2l.hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def get_data(name):
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    data_dir = os.path.join(base_dir, folder) if None else data_dir
    return data_dir

def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels


data_dir = get_data('SNLI')
train_data = read_snli(data_dir, is_train=True)
for p, h, l in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print("premise: ", p)
    print("hypotheses: ", h)
    print("label: ", l)


# Podemos ver que o dataset esta bem balanceado entre as classes
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])

# O dataset deve ter o mesmo tamanho no minibatch
# num_steps especifica o tamanho desejado
# Sentenças maiores serao subdimensionada
# e maiores serão adicionadas tokens <pad>

class SNLIDataset(torch.utils.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5,
                                   reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([
            d2l.truncate_pad(self.vocab[line], self.num_steps,
                             self.vocab['<pad>']) for line in lines
        ])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)


# Colocando tudo junto
# Retorna dataset instanciado como DataLoader
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = get_data('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True, num_workers = 0) 
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False, num_workers = 0)
    return train_iter, test_iter, train_set.vocab


train_iter, test_iter, vocab = load_data_snli(128, 50)



# Vamos ver como esta os dados
# As entradas X[0] e X[1] representa os pares de sentence
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break


## Definindo mlp para computar attention weight
def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)

# Camada de Attend
# Essa camada retorna Beta e Alpha

class Attend(nn.Module):
    def __init__(self, num_input, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_input, num_hiddens, flatten = False)
        
    def forward(self, A, B):
        # Formato A e B (batch, num_word, embed_size)
        # Formato f_A, f_B (batch, num_word, num_hiddens)
        f_A = self.f(A)
        f_B = self.f(B)
        # Formato e (batch, num_word_a, num_word_b)
        e = torch.bmm(f_A, f_B.permute(0,2,1))
        # Formato beta (batch, num_word_a, embed_size)
        beta = torch.bmm(F.softmax(e, dim = -1), B)
        # Formato alpha (batch, num_word_b, embed_size)
        alpha = torch.bmm(F.softmax(e.permute(0,2,1), dim = -1), A)
        return beta, alpha


# Camada de Comparing
class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten = False)
        
    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat([A, beta], dim = 2))
        V_B = self.g(torch.cat([B, alpha], dim = 2))
        return V_A, V_B


# Camada de Aggregate
class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten = False) 
        self.linear = nn.Linear(num_hiddens, num_outputs)
        
    def forward(self, V_A, V_B):
        V_A = V_A.sum(dim = 1)
        V_B = V_B.sum(dim = 1)
        Y_hat = self.linear(self.h(torch.cat([V_A,V_B], dim = 1)))
        return Y_hat

## Modelo DecomposableAttention
class DecomposableAttention(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens,
                 num_inputs_attend = 100, num_inputs_compare = 200,
                 num_inputs_agg = 400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend,num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)
        
    def forward(self, X):
        premises, hypothesis = X
        A = self.embedding(premises)
        B = self.embedding(hypothesis)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat


batch_size, num_steps = 256, 50
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)

# Download pré-treinado GloVe para embedding
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

glove_embedding = TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds);

lr , num_epochs = 0.001, 5
trainer = torch.optim.Adam(net.parameters(), lr= lr)
loss = nn.CrossEntropyLoss(reduction='none')
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
