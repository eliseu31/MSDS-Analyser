from data_extraction.file_interpreter import FileInterpreter
from data_management.text_pipeline import TextPipeline
from data_management.bootstrap import BootstrapLabels


if __name__ == '__main__':
    # get the data from the web
    # scraper.DataScraper()()

    # extract the text from the files
    df_dict = FileInterpreter(data_folder='small_datasheets')()
    # describe each dataframe
    for df_name, df in df_dict.items():
        print('\n\n#### DATAFRAME', df_name, '####')
        df.info()

    # passes the textual tokens to the text manager
    text_pipe = TextPipeline(df_dict=df_dict)

    # search in the train df
    # bootstrap = BootstrapLabels()
    # text_pipe.x_train.apply(bootstrap.search_pattern)
    # print('patterns set size', len(bootstrap.patterns_founded))
    # # print(list(bootstrap.patterns_founded)[5])
    # text_pipe.x_train.apply(bootstrap.search_tuple)
    # print(len(bootstrap.tuples_founded))

    text_pipe.text_pipelines()
    text_pipe.fit()
    text_pipe.score()
