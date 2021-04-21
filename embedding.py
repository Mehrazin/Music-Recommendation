import pandas as pd
import numpy as np
import scipy
import pickle
import nltk
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def plot_tsne(input_vectors, labels, artist_names, n_iter=2000):
    tsne = TSNE(n_components=2, n_iter=n_iter, init='pca')
    reduced_vectors = tsne.fit_transform(input_vectors)
    scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels)
    plt.legend(handles=scatter.legend_elements()[0], labels=artist_names)
    plt.show()


def get_artist_vectors(df, vectors, artist_names):
    labels = []
    all_vectors = []
    for i, name in enumerate(artist_names):
        artist_vectors = vectors[(df['artist_name'] == name).values]
        if isinstance(vectors, scipy.sparse.csr.csr_matrix):
            artist_vectors = artist_vectors.todense()
        all_vectors.append(artist_vectors)
        labels += all_vectors[-1].shape[0] * [i]

    all_vectors = np.concatenate(all_vectors, axis=0)
    return all_vectors, labels


class TFIDFEmbedding:
    def __init__(self, data, config):
        self.base_path = config.exp_dir
        df = data
        df = df.dropna(axis=0, how='any')
        df['lyrics'] = df['lyrics'].apply(lambda x: x.replace("\n", " "))
        self.df = df

        if os.path.exists(os.path.join(base_path, 'tfidf_vectors.pkl')):
            with open(os.path.join(base_path, 'tfidf_vectors.pkl'), 'rb') as f:
                self.tfidf_vectors = pickle.load(f)
        else:
            self.tfidf_vectors = self.tfidf(min_df=5, max_features=config.tfidf_vector_size)
            with open(os.path.join(self.base_path, 'tfidf_vectors.pkl'), 'wb') as f:
                pickle.dump(self.tfidf_vectors, f)

        self.merge_save()

    def tfidf(self, min_df, max_features):
        tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words,
                                           lowercase=True,
                                           tokenizer=word_tokenize,
                                           min_df=min_df,
                                           max_features=max_features)
        tfidf_vectors = tfidf_vectorizer.fit_transform(self.df['lyrics'])
        return tfidf_vectors

    def merge_save(self):
        new_df = self.df.copy(deep=True)
        new_df['embedding'] = [vector for vector in self.tfidf_vectors]
        new_df = new_df.drop(['lyrics'], axis=1)
        new_df.to_csv(os.path.join(self.base_path, 'df_tfidf.csv'))


class Doc2VecEmbedding:
    def __init__(self, data, config):
        self.base_path = config.exp_dir
        df = data
        df = df.dropna(axis=0, how='any')
        df['lyrics'] = df['lyrics'].apply(lambda x: x.replace("\n", " "))
        self.df = df

        if os.path.exists(os.path.join(base_path, 'doc2vec_vectors.pkl')):
            with open(os.path.join(base_path, 'doc2vec_vectors.pkl'), 'rb') as f:
                self.doc2vec_vectors = pickle.load(f)
        else:
            self.tagged_docs = self.tag_documents()
            self.model = self.doc2vec_train(vector_size = config.doc2vec_vector_size, epochs = config.doc2vec_epoochs)
            self.doc2vec_vectors = self.infer()
            with open(os.path.join(self.base_path, 'doc2vec_vectors.pkl'), 'wb') as f:
                pickle.dump(self.doc2vec_vectors, f)

        self.merge_save()

    def tag_documents(self, save=True, file_name='tagged_docs'):
        print("tagging documents")
        path = os.path.join(self.base_path, file_name + '.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                tagged_docs = pickle.load(f)
        else:
            tagged_docs = [TaggedDocument(
                words=[w for w in word_tokenize(doc.lower()) if w not in stop_words],
                tags=[str(i)]) for i, doc in tqdm(enumerate(self.df.lyrics), total=len(self.df))]
            if save:
                with open(path, 'wb') as f:
                    pickle.dump(tagged_docs, f)
        return tagged_docs

    def doc2vec_train(self, vector_size=100,
                      epochs=50, alpha=0.025, min_alpha=0.00025,
                      min_count=5, save=True, model_name='doc2vec'):

        print("training doc2vec")
        path = os.path.join(self.base_path, model_name + '.model')
        if os.path.exists(path):
            model = Doc2Vec.load(path)
        else:
            model = Doc2Vec(vector_size=vector_size,
                            epochs=epochs,
                            alpha=alpha,
                            min_alpha=min_alpha,
                            min_count=min_count,
                            dm=1)
            model.build_vocab(self.tagged_docs)
            model.train(self.tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
            if save:
                model.save(path)
        return model

    def infer(self, save=True, file_name='doc2vec_vectors'):
        print("inferring vectors")
        path = os.path.join(self.base_path, file_name + '.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                vectors = pickle.load(f)
        else:
            vectors = np.stack([self.model.infer_vector(tagged_doc.words) for tagged_doc in tqdm(self.tagged_docs)],
                               axis=0)
            if save:
                with open(os.path.join(self.base_path, file_name + '.pkl'), 'wb') as f:
                    pickle.dump(vectors, f)
        return vectors

    def merge_save(self):
        new_df = self.df.copy(deep=True)
        new_df['embedding'] = [vector for vector in self.doc2vec_vectors]
        new_df = new_df.drop(['lyrics'], axis=1)
        new_df.to_csv(os.path.join(self.base_path, 'df_doc2vec.csv'))


if __name__ == "__main__":
    root = './'
    tfidf = TFIDFEmbedding(root)
    doc2vec = Doc2VecEmbedding(root, vector_size=100)
