from keras.layers import Dense, Input
from keras.models import Sequential
from prototype.SDAE.data_handler import Sentences
from gensim.models.word2vec import Word2Vec
import os

cur_dir = os.path.dirname(__file__)
project_root = os.path.join(cur_dir, '..', '..')
model_path = os.path.join(project_root, 'models', 'nlp')
w2v_model_path = os.path.join(model_path, 'w2v.txt')
data_path = os.path.join(project_root, 'data', 'NLP', 'books_large_p1.txt')




def main():

    c = 0
    for sentence in Sentences.from_file(data_path):
        print(sentence)
        c += 1
        if c == 100:
            exit()


    w2v = Word2Vec.load_word2vec_format(w2v_model_path, binary=False)
    ret = w2v.similar_by_word('word')
    print(ret)

if __name__ == '__main__':
    main()