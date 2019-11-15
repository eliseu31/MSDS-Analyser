from data_extraction.file_interpreter import FileInterpreter
from data_management.text_converter import TextMatrix


if __name__ == '__main__':
    # get the data from the web
    # scraper.DataScraper()()

    # extract the text from the files
    df_dict = FileInterpreter(data_folder='small_datasheets')()
    # describe each dataframe
    for df_name, df in df_dict.items():
        print('\n\n#### DATAFRAME:', df_name)
        print('#### INFO:')
        df.info()
        print('#### DESCRIPTION:')
        df.describe().transpose()

    # passes the textual tokens to the text manager
    text_matrix = TextMatrix(df_dict=df_dict)()
