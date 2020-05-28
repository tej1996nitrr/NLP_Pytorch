#%%
#TF Representation
# the sum of the onehot representations of its constituent words.
#%%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import torch
import numpy as np
#%% 
corpus = ['Time flies flies like an arrow.',
'Fruit flies like a banana.']
vocab=set()
for sentence in corpus:
    for word in sentence.split(" "):
        vocab.add(word.lower())
print(list(vocab))
vocab=list(vocab)
one_hot_vectorizer = CountVectorizer(binary=True)
one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()
sns.heatmap(one_hot, annot=True,cbar=False,xticklabels=vocab,yticklabels=['Sentence 2'])

# %%
#-----------------------------TF-IDF------------------------
# The IDF representation penalizes common tokens and rewards rare tokens in the vector representation
"""The IDF(w) of a token w is defined with respect to a corpus as:
IDF(w)=log (N/n)
where n is the number of documents containing the word w and N is the total number of documents.
The TFIDF score is simply the product TF(w) * IDF(w). 
First, notice how if there is a very common
word that occurs in all documents (i.e., n = N), IDF(w) is 0 and the TFIDF
score is 0, thereby
completely penalizing that term. 
Second, if a term occurs very rarely, perhaps in only one document,
the IDF will be the maximum possible value, log N.
E
xample 12"""

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus).toarray()
sns.heatmap(tfidf, annot=True, cbar=False, xticklabels=vocab,
yticklabels= ['Sentence 1', 'Sentence 2'])


# %%
#----------------------------Creating Tensors-----------------------
def describe(x):
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))

# %%

describe(torch.Tensor(2))
describe(torch.rand(2, 3)) # uniform random
describe(torch.randn(2, 3)) #random normal
describe(torch.zeros(2,3))
#%%
x = torch.ones(2, 3)
describe(x)
x.fill_(5) #_=>inplace
describe(x)

# %%
#---------------------------tensor using lists-------------------------
x = torch.Tensor([[1, 2, 3],[4, 5, 6]])
describe(x)

# %%
#---------------------------tensor using numpy-------------------------

npy = np.random.rand(2, 3)
describe(torch.from_numpy(npy))

# %%
#-------------------------------tensor operations--------------------------
x = torch.randn(2, 3)
describe(x)
describe(torch.add(x, x))
describe(x + x)

# %%
x = torch.arange(6)
describe(x)

# %%
x = x.view(2, 3)
describe(x)

# %%
describe(torch.sum(x, dim=1))

# %%
describe(torch.transpose(x, 0, 1))

# %%
#----------------------Indexing/slicing------------------------------
x = torch.arange(6).view(2, 3)
describe(x)

# %%
describe(x[:2, :2])
describe(x[:1, :2])

# %%
for i in range(x.shape[0]+1):
    for j in range(x.shape[1]+1):
        describe(x[:i,:j])

# %%
indices = torch.LongTensor([0, 2])
describe(torch.index_select(x, dim=1, index=indices))

# %%
indices = torch.LongTensor([0, 1])
describe(torch.index_select(x, dim=0, index=indices))

# %%
row_indices = torch.arange(2).long()
col_indices = torch.LongTensor([0, 1])
describe(x[row_indices, col_indices])

# %%
