import gensim

sentences = [['AI', 'prac', 'first', 'sentence'], ['AI', 'prac', 'second', 'sentence']]
model = gensim.models.Word2Vec(sentences, min_count=1)
print(model)

words = list(model.wv.vocab)
print(words)

