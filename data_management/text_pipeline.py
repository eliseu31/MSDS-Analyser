from data_management.visualization import PlotGraphics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

INPUT_FEATURES = ['health', 'chemical', 'personal']
OUTPUT_TARGETS = ['procedures']


class TextPipeline:

    def __init__(self, df_dict):
        data_dict = {}
        for name, data in [('feature', INPUT_FEATURES), ('target', OUTPUT_TARGETS)]:
            # join all the data frames
            df_list = [df_value for df_name, df_value in df_dict.items()
                       if df_name in data]
            df = pd.concat(df_list, axis=1)
            # join all the strings in the dataframe
            data_dict[name] = df.apply(' '.join, axis=1)
        # split the train and test dataset
        data_tuple = train_test_split(data_dict['feature'], data_dict['target'], test_size=0.30, shuffle=True)
        self.x_train, self.x_test, self.y_train, self.y_test = data_tuple

        # text extraction pipelines
        self.text_extraction_pipes = {}
        # prediction model to use
        self.prediction_model = DecisionTreeClassifier(random_state=0)
        # init the visualizer
        self.plt_graphics = PlotGraphics(data_tuple, self.text_extraction_pipes)

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

        # plot the pca
        # class_labels = self.text_extraction_pipes['target']['vectorizer'].get_feature_names()
        # self.plot_pca(transform_data[0], transform_data[1], class_labels)

        # convert the y_train
        y_data = transform_data[1].toarray()
        y_data[y_data > 1] = 1
        # fit the model
        self.prediction_model.fit(transform_data[0], y_data)

    def score(self):
        # score the model
        x_test_vector = self.text_extraction_pipes['feature'].transform(self.x_test)
        y_test_vector = self.text_extraction_pipes['target'].transform(self.y_test).toarray()
        # convert the y_test
        predictions = self.prediction_model.predict(x_test_vector)
        # print some metrics
        class_labels = self.text_extraction_pipes['target']['vectorizer'].get_feature_names()
        class_report = self.calculate_metrics(y_test_vector, predictions, class_labels)
        # plot the data
        self.plt_graphics.plot_bag_words(class_report)

    @staticmethod
    def calculate_metrics(y_test, predictions, class_labels):
        # print the results
        y_test[y_test > 1] = 1
        class_report = classification_report(y_test, predictions, target_names=class_labels, output_dict=True)
        print("Classification report: \n", classification_report(y_test, predictions, target_names=class_labels))
        print("F1 micro averaging:", f1_score(y_test, predictions, average='micro', labels=np.unique(predictions)))
        print("ROC: ", roc_auc_score(y_test, predictions))
        # return the classification results
        return class_report
