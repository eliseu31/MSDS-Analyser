from core.text_pipeline import TextPipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import joblib
import os
import sys


class PipelinesManager:

    def __init__(self, df_dict=None):
        # root path
        self.resources_folder = os.path.join(os.path.dirname(sys.path[0]), 'resources')
        # parses the data
        if df_dict is not None:
            # pickles the data
            self.pickle_data(df_dict)
        else:
            # unpickle the data
            df_dict = self.unpickle_data()

        # describe each dataframe
        for df_name, df in df_dict.items():
            print('\n\n#### DATAFRAME', df_name, '####')
            df.info()

        # join all the features data frames
        df = pd.concat([df_dict['health'], df_dict['chemical'], df_dict['personal']], axis=1)
        features = df.apply(' '.join, axis=1)
        # split the train and test dataset
        data_tuple = train_test_split(features, df_dict['procedures'], test_size=0.30, shuffle=True, random_state=11)
        self.x_train, self.x_test, self.y_train, self.y_test = data_tuple

        # dict to store the pipelines
        self.pipelines_dict = dict()
        # features pipeline object
        self.features_pipeline = None

    def create_predictors(self, procedures_list, relation_extraction=False,
                          n_features=150, n_targets=20, classifier='decision_tree', n_jobs=8):
        # create all the pipelines
        for pipeline_name, target_procedure in procedures_list:
            # join all the target data frames
            y_train = self.y_train[target_procedure].apply(' '.join, axis=1)
            y_test = self.y_test[target_procedure].apply(' '.join, axis=1)
            # generates the data tuple
            data_tuple = (self.x_train, self.x_test, y_train, y_test)
            # creates the text pipeline
            self.pipelines_dict[pipeline_name] = TextPipeline(pipeline_name, data_tuple, classifier_name=classifier)
            # creates the targets pipeline
            self.pipelines_dict[pipeline_name].create_label_pipeline(relation_extraction, n_targets, n_jobs)
            # creates the pipeline
            if self.features_pipeline is None:
                # creates the pipeline obj
                self.features_pipeline = self.pipelines_dict[pipeline_name].create_features_pipeline(n_features)
            else:
                # passes the object
                self.pipelines_dict[pipeline_name].create_features_pipeline(pipeline_obj=self.features_pipeline)

    def pickle_data(self, df_dict):
        # pickle the dataset
        joblib.dump(df_dict['health'], os.path.join(self.resources_folder, 'df_health.pkl'))
        joblib.dump(df_dict['chemical'], os.path.join(self.resources_folder, 'df_chemical.pkl'))
        joblib.dump(df_dict['personal'], os.path.join(self.resources_folder, 'df_personal.pkl'))
        joblib.dump(df_dict['procedures'], os.path.join(self.resources_folder, 'df_procedures.pkl'))

    def pickle_system(self):
        # pickle the pipelines
        for _, pipeline in self.pipelines_dict.items():
            # pickle each pipeline
            pipeline.pickle_predictor()
        # pickle the features pipeline
        obj_name = os.path.join(self.resources_folder, 'features_vectorizer')
        joblib.dump(self.features_pipeline['vectorizer'], obj_name + '.pkl')

    def unpickle_data(self):
        # initializes the dict
        df_dict = dict()
        # unpickle the dataset
        df_dict['health'] = joblib.load(os.path.join(self.resources_folder, 'df_health.pkl'))
        df_dict['chemical'] = joblib.load(os.path.join(self.resources_folder, 'df_chemical.pkl'))
        df_dict['personal'] = joblib.load(os.path.join(self.resources_folder, 'df_personal.pkl'))
        df_dict['procedures'] = joblib.load(os.path.join(self.resources_folder, 'df_procedures.pkl'))
        # returns the dfs
        return df_dict

    def unpickle_system(self, procedures_list):
        # unpickle the features pipeline
        obj_name = os.path.join(self.resources_folder, 'features_vectorizer')
        features_vectorizer = joblib.load(obj_name + '.pkl')
        self.features_pipeline = Pipeline([('vectorizer', features_vectorizer)])

        # unpickle the pipelines
        for pipeline_name, target_procedure in procedures_list:
            # join all the target data frames
            train_targets = self.y_train[target_procedure].apply(' '.join, axis=1)
            test_targets = self.y_test[target_procedure].apply(' '.join, axis=1)
            # generates the data tuple
            data_tuple = (self.x_train, self.x_test, train_targets, test_targets)
            # creates the text pipeline
            self.pipelines_dict[pipeline_name] = TextPipeline(pipeline_name, data_tuple)
            # unpickle their pipelines
            self.pipelines_dict[pipeline_name].unpickle_predictor()
            # passes the features pipeline
            self.pipelines_dict[pipeline_name].create_features_pipeline(pipeline_obj=self.features_pipeline)

    def fit(self):
        # fits the features pipeline
        x_vector = self.features_pipeline.fit_transform(self.x_train)
        # iterate over all pipelines
        for name, pipeline in self.pipelines_dict.items():
            # fit each one
            pipeline.fit(x_vector)

    def predict(self, x_data):
        # filter the dict features
        features_categories = ['health', 'chemical', 'personal']
        features_iter = list(filter(lambda x: (x[0] in features_categories) and (x[1] is not None), x_data.items()))
        # prints the extracted data
        print('\n#### EXTRACTED DATA ####')
        for f_category, data in features_iter:
            print('\nCATEGORY:', f_category)
            for key, value in data.items():
                print('{0}: {1}'.format(key, value))
        # join all the features
        x_str = ' '.join([' '.join(filter(lambda x: x is not None, data.values()))
                          for _, data in features_iter])
        print('\n#### PREDICTIONS ####')
        # iterate over all pipelines
        for name, pipeline in self.pipelines_dict.items():
            # predicts the result to each pipeline
            y_array = pipeline.predict(np.array([x_str]))
            # converts the array to labels
            labels = zip(pipeline.text_extraction_pipes['target']['vectorizer'].get_feature_names(), y_array[0])
            filtered_labels = list(filter(lambda x: x[1] != 0, labels))
            # print the procedures
            print('\nPREDICTED PROCEDURES:', name)
            if len(filtered_labels) > 0:
                labels, _ = zip(*filtered_labels)
                # prints the output
                print(list(labels))

    def score(self):
        # iterate over all pipelines
        for name, pipeline in self.pipelines_dict.items():
            # score each one
            pipeline.score()
