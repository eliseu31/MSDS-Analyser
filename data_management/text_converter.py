from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

INPUT_FEATURES = ['health', 'chemical', 'personal']
OUTPUT_TARGETS = ['procedures']


class TextMatrix:

    def __init__(self, df_dict=None, files_path=None):
        # read from the file
        if df_dict is None:
            # read from the dataframe
            pass

        data_dict = {}
        for name, data in [('feature', INPUT_FEATURES), ('target', OUTPUT_TARGETS)]:
            # join all the data frames
            df_list = [df_value for df_name, df_value in df_dict.items()
                       if df_name in data]
            df = pd.concat(df_list, axis=1)
            # join all the strings in the dataframe
            data_dict[name] = df.apply(' '.join, axis=1)
        # split the train and test dataset
        split_tuple = train_test_split(data_dict['feature'], data_dict['target'], test_size=0.30, shuffle=True)
        self.x_train, self.x_test, self.y_train, self.y_test = split_tuple

        # text extraction pipelines
        self.text_extraction_pipes = {}
        # prediction model to use
        self.prediction_model = DecisionTreeClassifier(random_state=0)

    def text_pipelines(self):
        # create the 2 different pipes
        for pipe_name, n_features in [('feature', 150), ('target', 30)]:
            # vectorization
            count_vectorizer = CountVectorizer(strip_accents='unicode',
                                               stop_words='english',
                                               lowercase=True,
                                               max_features=n_features,
                                               ngram_range=(1, 2),
                                               min_df=0.1, max_df=0.7)
            # list with the steps
            pipe_steps = [('vectorizer', count_vectorizer)]
            # to the feature pipeline add the tf-idf transform
            if pipe_name == 'feature':
                # transforms and normalizes
                tfid_transformer = TfidfTransformer()
                # add to the steps
                pipe_steps.append(('transformer', tfid_transformer))

            # pipeline creation
            self.text_extraction_pipes[pipe_name] = Pipeline(pipe_steps)

    def fit(self):
        transform_data = []
        # fit the 2 different pipes
        for name, x_data in [('feature', self.x_train), ('target', self.y_train)]:
            # fits the pipeline
            trans_data = self.text_extraction_pipes[name].fit_transform(x_data)
            transform_data.append(trans_data)

            # print some information data
            print('\nactual pipe:', name)
            print('input array shape:', x_data.shape, type(x_data))
            print('output array shape:', trans_data.shape, type(trans_data))
            # plot word frequency
            count_vectorizer = self.text_extraction_pipes[name]['vectorizer']
            self.plot_word_freq(count_vectorizer, x_data)

        # plot the pca
        # class_labels = self.text_extraction_pipes['target']['vectorizer'].get_feature_names()
        # self.plot_pca(transform_data[0], transform_data[1], class_labels)

        # convert the y_train
        y_data = transform_data[1].toarray()
        y_data[y_data > 1] = 1
        print(y_data)
        # fit the model
        self.prediction_model.fit(transform_data[0], y_data)
        # score the model
        x_test_vector = self.text_extraction_pipes['feature'].transform(self.x_test)
        y_test_vector = self.text_extraction_pipes['target'].transform(self.y_test).toarray()
        # convert the y_test
        y_test_vector[y_test_vector > 1] = 1
        predictions = self.prediction_model.predict(x_test_vector)
        # print the results
        print(multilabel_confusion_matrix(y_test_vector, predictions))
        print(accuracy_score(y_test_vector, predictions))

    @staticmethod
    def plot_word_freq(vectorizer, string_df):
        # plot the vocabulary freq
        bag_of_words = vectorizer.transform(string_df)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, i]) for word, i in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        words, total = zip(*words_freq)
        plt.bar(words[0:60], total[0:60])
        plt.xticks(rotation=90)
        plt.show()

    @staticmethod
    def plot_pca(x_data, y_data, class_labels):
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(x_data.todense())
        print(pca_data)
        print(pca_data.shape)
        print(y_data.shape)
        print(y_data.todense())

        # plot the pca
        # for class_label in class_labels:
        #     plt.scatter(pca_data[:, 0], pca_data[:, 1], c=y_data)
        #     plt.show()
