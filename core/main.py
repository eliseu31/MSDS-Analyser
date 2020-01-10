import argparse
import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from extraction.file_interpreter import FileInterpreter
from extraction.scraper import DataScraper
from core.pipelines_manager import PipelinesManager

if __name__ == '__main__':
    df_dict = None
    n_jobs = 8
    data_folder = 'datasheets_smaller'
    classifier = 'decision_tree'
    n_features = 150
    n_targets = 20

    procedures_map = {'fire': ('fire_fighting', ['extinguishing_media', 'fire_fighting']),
                      'disposal': ('disposal', ['disposal']),
                      'storage': ('storage', ['handling_storage'])}
    procedures2use = ['fire']

    # build parser for application command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', metavar='n_jobs', nargs=1, help="number of jobs to use (default: 8)")
    parser.add_argument('-s', action='store_true', help="scrapes the MSDS data from the web")
    parser.add_argument('-e', metavar='data_folder', nargs=1, help="extract the dataframes from the dataset, you "
                                                                   "must specify the dataset name e.g. datasheets")
    parser.add_argument('-t', metavar=('n_feature', 'n_class'), nargs=2,
                        help="train with 70%% of the dataset and score with 30%% of the dataset, specify the number of "
                             "features and target classes (default 150 features and 20 classes)")
    parser.add_argument('-l', metavar='procedures_list', nargs=1,
                        help="""procedures to use as labels (e.g. '["fire", "storage", "disposal"]')""")
    parser.add_argument('-c', metavar='classifier model', nargs=1,
                        help="classifier model to use (decision_tree or k_neighbors)")
    parser.add_argument('-p', metavar='file_path', nargs=1,
                        help="predict the procedures for a certain MSDS with a file path")
    args = parser.parse_args()

    if args.s:
        # get the data from the web
        DataScraper()()

    if args.e:
        # get the arguments
        data_folder = args.e[0]
        # extract the text from the files
        df_dict = FileInterpreter(data_folder=data_folder)(n_jobs=n_jobs)
        # to pickle data
        PipelinesManager(df_dict)

    if args.l:
        # get the procedures
        procedures2use = json.loads(args.l[0])

    # maps the procedures list
    procedures_list = [data_item for k, data_item in procedures_map.items() if k in procedures2use]

    if args.c:
        # get the classifier
        classifier = args.c[0]

    if args.t:
        # get the arguments
        n_features = int(args.t[0])
        n_targets = int(args.t[1])
        # passes the textual tokens to the text manager
        pipes_manager = PipelinesManager(df_dict)
        pipes_manager.create_predictors(procedures_list,
                                        relation_extraction=True,
                                        n_features=n_features,
                                        n_targets=n_targets,
                                        classifier=classifier,
                                        n_jobs=n_jobs)
        pipes_manager.fit()
        pipes_manager.score()
        pipes_manager.pickle_system()

    if args.p:
        # creates the pipes manager
        pipes_manager = PipelinesManager()
        pipes_manager.unpickle_system(procedures_list)

        # get the arguments
        file_path = args.p[0]
        absolute_path = os.path.join(os.path.dirname(sys.path[0]), file_path)
        # extracts the data
        with open(file_path) as f:
            text = f.read()
            # extracts the data
            file_data = FileInterpreter().process_f2(text)

        # predicts the file procedures
        pipes_manager.predict(file_data)
