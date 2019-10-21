from data_management import scraper
from data_management import text_extractor
from data_management import text_processing


if __name__ == '__main__':
    # get the data from the web
    # scraper.DataScraper()()

    # extract the text from the files
    df = text_extractor.TextExtractor(data_folder='small_datasheets')()



