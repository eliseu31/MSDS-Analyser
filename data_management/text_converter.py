from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

INPUT_FEATURES = ['health', 'chemical', 'personal']
OUTPUT_TARGETS = ['procedures']


class TextMatrix:

    def __init__(self, df_dict=None, files_path=None):
        # stores the 5 df's
        self.df_dict = df_dict
        self.pipelines = {}

        # read from the file
        if self.df_dict is None:
            # read from the dataframe
            pass

    def __call__(self):
        # pca data
        pca_data = {}
        # iterate over all the dicts df
        for df_name in INPUT_FEATURES + OUTPUT_TARGETS:
            # join all the strings in the dataframe
            joined_df = self.df_dict[df_name].apply('. '.join, axis=1)
            # vectorization
            count_vectorizer = CountVectorizer(lowercase=True)
            # transforms and normalizes
            tfid_transformer = TfidfTransformer()
            # pipeline creation
            self.pipelines[df_name] = Pipeline([('vectorizer', count_vectorizer),
                                                ('transformer', tfid_transformer)])
            # fits the pipeline
            visualization_data = self.pipelines[df_name].fit_transform(joined_df).todense()
            pca_data[df_name] = visualization_data

        # target data oca
        pca_target = PCA(n_components=1)
        output_data = pca_target.fit_transform(pca_data['procedures'])
        # feature data pca
        pca_features = PCA(n_components=2)
        features_data = pca_features.fit_transform(pca_data['health'])
        # plot the data
        plt.scatter(features_data[:, 0], features_data[:, 1], c=output_data[:, 0])
        plt.show()

        # embedded layer
