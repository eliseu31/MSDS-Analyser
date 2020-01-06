from bs4 import BeautifulSoup
import urllib3
import sys
import os


class DataScraper:
    http = urllib3.PoolManager()
    # define the root pages
    root_urls = {'f2': 'https://hazard.com/msds/f2/'}
    # other links
    # 'f1': 'https://hazard.com/msds/f1/',
    # 'mf': 'https://hazard.com/msds/mf/'

    def __call__(self, *args, **kwargs):
        # start scraping
        for name, root_url in self.root_urls.items():
            # gets the links and remove the first 5 links (table headers)
            parent_links = self._scrape_links(root_url)[60:]

            # folder to save the dataset
            root_folder = os.path.join(os.path.dirname(sys.path[0]), 'datasheets')
            # creates a folder parent
            parent_folder = self._create_folder(name, root_folder)

            # iterates over each link
            for parent_name in parent_links:
                # makes a new folder for each link
                files_folder = self._create_folder(parent_name, parent_folder)
                # scrape each data folder and remove the first 5 links (table headers)
                file_links = self._scrape_links(url=root_url+parent_name)[5:]
                print(root_url+parent_name)
                # scrape each file
                for file_name in file_links:
                    try:
                        print('downloading...', file_name)
                        # extract the data
                        file_data = self._extract_datasheet(url=root_url+parent_name+file_name)
                        # format the file name
                        file_name = file_name.split('.')[0] + '.txt'
                        file_path = os.path.join(files_folder, file_name)
                        # write to the file
                        with open(file_path, 'w') as text_file:
                            text_file.write(file_data)
                    except AttributeError as e:
                        print(e)
                        continue

    def _scrape_links(self, url):
        # get the root page
        root_page = self.http.request('GET', url)
        # find the links
        soup = BeautifulSoup(root_page.data, 'html.parser')
        # print(soup.prettify())
        links = []
        for link in soup.findAll('a'):
            links.append(link.get('href'))
        # returns the links list
        return links

    @staticmethod
    def _create_folder(folder_name, parent_folder):
        # create a folder to store the dataset
        new_folder = os.path.join(parent_folder, folder_name)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        return new_folder

    def _extract_datasheet(self, url):
        # get the root page
        root_page = self.http.request('GET', url)
        # find the links
        soup = BeautifulSoup(root_page.data, 'html.parser')
        file_text = soup.find('pre').get_text()
        # returns the text of the file
        return file_text
