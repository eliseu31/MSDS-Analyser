from core.visualization import PlotGraphics
from core.relation_extraction import SpacyRelationExtraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import numpy as np
import joblib
import os
import sys
import warnings

warnings.filterwarnings('ignore')


class TextPipeline:

    def __init__(self, predictor_name, data_tuple, classifier_name='decision_tree', resources_path=None):
        # root path
        self.resources_folder = os.path.join(os.path.dirname(sys.path[0]), 'resources') \
            if resources_path is None else resources_path
        # initializes the classifier dict
        classifiers = {'decision_tree': DecisionTreeClassifier(random_state=0),
                       'k_neighbors': KNeighborsClassifier(n_neighbors=15)}
        # save the predictor name
        self.predictor_name = predictor_name

        # receives the data
        self.x_train, self.x_test, self.y_train, self.y_test = data_tuple
        # text extraction pipelines
        self.text_extraction_pipes = {}
        # prediction model to use
        self.prediction_model = classifiers[classifier_name]
        # init the visualizer
        self.plt_graphics = PlotGraphics(data_tuple, self.text_extraction_pipes)

    def create_features_pipeline(self, n_features=150, pipeline_obj=None):
        # checks if the transformer
        if pipeline_obj is None:
            # features vectorization
            transformer_obj = TfidfVectorizer(strip_accents='unicode',
                                              stop_words='english',
                                              lowercase=True,
                                              max_features=n_features,
                                              ngram_range=(1, 2),
                                              min_df=0.1, max_df=0.7)
            # creates the pipeline obj
            pipeline_obj = Pipeline([('vectorizer', transformer_obj)])
        # pipeline mapping
        self.text_extraction_pipes['feature'] = pipeline_obj
        # returns the pipeline obj
        return self.text_extraction_pipes['feature']

    def create_label_pipeline(self, relation_extraction=False, n_targets=20, n_jobs=8):
        # target vectorization
        if relation_extraction:
            # uses the spacy relation extraction
            vectorizer = SpacyRelationExtraction(n_relation=n_targets, n_jobs=n_jobs)
        else:
            # otherwise uses a normal vectorizer
            vectorizer = CountVectorizer(strip_accents='unicode',
                                         stop_words='english',
                                         lowercase=True,
                                         max_features=n_targets,
                                         ngram_range=(1, 2),
                                         min_df=0.1, max_df=0.7)
        # pipeline creation
        self.text_extraction_pipes['target'] = Pipeline([('vectorizer', vectorizer)])

    def pickle_predictor(self):
        # save the labels pipeline
        labels_extractor = self.text_extraction_pipes['target']['vectorizer']
        obj_name = os.path.join(self.resources_folder, '_'.join([self.predictor_name, 'labels', 'vectorizer']))
        joblib.dump(labels_extractor, obj_name + '.pkl')

        # saves the model
        obj_name = os.path.join(self.resources_folder, '_'.join([self.predictor_name, 'predictor']))
        joblib.dump(self.prediction_model, obj_name + '.pkl')

    def unpickle_predictor(self):
        # loads the object
        obj_name = os.path.join(self.resources_folder, '_'.join([self.predictor_name, 'labels', 'vectorizer']))
        labels_extractor = joblib.load(obj_name + '.pkl')
        self.text_extraction_pipes['target'] = Pipeline([('vectorizer', labels_extractor)])

        # unpickle the model
        obj_name = os.path.join(self.resources_folder, '_'.join([self.predictor_name, 'predictor']))
        self.prediction_model = joblib.load(obj_name + '.pkl')

    def fit(self, x_vector):
        # fit the feature data
        y_vector = self.text_extraction_pipes['target'].fit_transform(self.y_train).toarray()
        # convert the y_train
        y_vector[y_vector > 1] = 1
        # print some information data
        print('\ninput array, shape:', x_vector.shape)
        print('output array, shape:', y_vector.shape, '\n')

        # fit the model
        self.prediction_model.fit(x_vector, y_vector)

    def predict(self, x_test):
        # convert using the pipeline
        x_test_vector = self.text_extraction_pipes['feature'].transform(x_test)
        # convert the y_test
        predictions = self.prediction_model.predict(x_test_vector)
        # returns the predictions
        return predictions

    def score(self):
        # add the exception treatment
        y_test_vector = self.text_extraction_pipes['target'].transform(self.y_test).toarray()
        # predict the output for the test set
        predictions = self.predict(self.x_test)

        # print some metrics
        class_labels = self.text_extraction_pipes['target']['vectorizer'].get_feature_names()
        class_report = self.calculate_metrics(y_test_vector, predictions, class_labels)
        # plot the data
        self.plt_graphics.plot_bag_words(class_report)

        # return the classification report
        return class_report

    @staticmethod
    def calculate_metrics(y_test, predictions, class_labels):
        # print the results
        y_test[y_test > 1] = 1
        class_report = classification_report(y_test, predictions, target_names=class_labels, output_dict=True)
        print("Classification report: \n", classification_report(y_test, predictions, target_names=class_labels))
        # print("F1 micro averaging:", f1_score(y_test, predictions, average='micro', labels=np.unique(predictions)))
        print("ROC: ", roc_auc_score(y_test, predictions), '\n')
        # return the classification results
        return class_report
