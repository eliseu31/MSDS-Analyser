from joblib import delayed, Parallel
import pandas as pd
import numpy as np
import nltk
import re
import os
import sys

# numerical data
IDENTIFICATION_DATA = ['product_name', 'company']
NUMERICAL_FEATURES = ['boiling_point', 'vapor_density', 'vapor_pressure', 'flash_point']
BOOL_FEATURES = ['carcinogenicity_iarc', 'carcinogenicity_ntp', 'carcinogenicity_osha']

# textual data
PROCEDURES_DESCRIPTION_FEATURES = ['spill', 'handling_storage', 'disposal', 'fire_fighting', 'extinguishing_media']
PERSONAL_DESCRIPTION_FEATURES = ['respiratory_protection', 'protective_equipment', 'other_precautions',
                                 'protective_gloves', 'eyes_protection', 'ventilation']
HEALTH_DESCRIPTION_FEATURES = ['health_hazards', 'effects_overexposure', 'first_aids']
CHEMICAL_DESCRIPTION_FEATURES = ['appearance', 'conditions_avoid', 'decomposition_products']


class FileInterpreter:

    def __init__(self, data_folder='datasheets'):
        # folder to save the dataset
        self.root_folder = os.path.join(os.path.dirname(sys.path[0]), data_folder)

    def __call__(self, save_df=False, n_jobs=8):
        folders = {'f2': self.process_f2}
        # iterates over each parent folder
        for folder_name, folder_method in folders.items():
            folder_path = os.path.join(self.root_folder, folder_name)
            child_folders = os.listdir(folder_path)
            # split in N equal slices
            chunks_size = len(child_folders) // n_jobs
            chunks = [child_folders[chunks_size * i: chunks_size * (i + 1)] for i in range(n_jobs)]
            chunks[n_jobs - 1] = child_folders[chunks_size * (n_jobs - 1):]
            # uses parallel processes
            with Parallel(n_jobs=n_jobs) as parallel:
                # process the files
                process_l = parallel(delayed(self.process_files)(folder_name, folder_method, chunk) for chunk in chunks)
                # join all the df's
                df_dict = {'numerical': [], 'health': [], 'personal': [],
                           'chemical': [], 'procedures': []}
                # join all the processes output
                for process_dict in process_l:
                    # iterate over each dict
                    for feature_value, value in process_dict.items():
                        # append to each feature list
                        df_dict[feature_value].append(value)

                for df_name, df_list in df_dict.items():
                    # zip all the results
                    df = pd.concat(df_list, sort=True)
                    df.set_index('file_name', inplace=True)
                    # replace the Nones with ''
                    df.fillna(value='', inplace=True)
                    # save the dataset
                    if save_df:
                        df.to_csv(self.root_folder, '{0}_data.csv'.format(df_name))
                    # stores the value
                    df_dict[df_name] = df

                # return the 5 dfs
                return df_dict

    def process_files(self, folder_name, method, chunk):
        df_dict = {'numerical': None, 'health': None, 'personal': None,
                   'chemical': None, 'procedures': None}
        # iterate over each folder
        for folder in chunk:
            folder_path = os.path.join(self.root_folder, folder_name, folder)
            # list all the file inside the folder
            file_names = os.listdir(folder_path)
            # iterate over each file
            for file_name in file_names:
                file_path = os.path.join(folder_path, file_name)
                # print(file_path)
                with open(file_path) as f:
                    text = f.read()
                    new_data_dict = method(text)
                    # get the file name
                    f_name = file_name.split('.')[0]
                    # update each df
                    for feature_name, new_data_item in new_data_dict.items():
                        # check if there are any data
                        if new_data_item is None:
                            continue
                        # append the file name to the dicts
                        new_data_dict[feature_name]['file_name'] = f_name
                        # build the ingredients df
                        if df_dict[feature_name] is None:
                            # create the df
                            df_dict[feature_name] = pd.DataFrame(columns=new_data_item.keys())
                        # append a row to the df
                        df_dict[feature_name] = df_dict[feature_name].append(new_data_item,
                                                                             ignore_index=True)
            print('processed', chunk.index(folder), 'of', len(chunk))

        # resets the index
        return df_dict

    def process_f2(self, text):
        # identification data
        product_name = re.search(r'^Product ID:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        company_info = re.findall(r'^=+\s+(.*)\s+=+\nCompany Name:(.+)$', text, re.M)
        _, company_info = company_info[0] if len(company_info) > 0 else (None, None)
        # store the identification information
        mapped_data = map(self.clean_name, [self.group_extraction(product_name), company_info])
        identification_data = dict(zip(IDENTIFICATION_DATA, mapped_data))

        # ******************* TEXTUAL DATA *******************

        # health description: process as list of things (eyes, ...)
        health_results = [re.search(r'^Health Hazards Acute and Chronic:((.*|\n)+(?!\n\s{3,}))', text, re.M),
                          re.search(r'^Effects of Overexposure:((.*|\n)+(?!\n\s{3,}))', text, re.M),
                          re.search(r'^First Aid:((.*|\n)+(?!\n\s{3,}))', text, re.M)]
        mapped_data = map(self.clean_health_description, HEALTH_DESCRIPTION_FEATURES, health_results)
        # pre process the health data
        health_data = {}
        for new_dict in list(mapped_data):
            health_data.update(new_dict)

        # description data (full texts)
        procedures_results = [re.search(r'^Spill Release Procedures:((.*|\n)+(?!\n\s{3,}))', text, re.M),
                              re.search(r'^Handling and Storage Precautions:((.*|\n)+(?!\n\s{3,}))', text, re.M),
                              re.search(r'^Waste Disposal Methods:((.*|\n)+(?!\n\s{3,}))', text, re.M),
                              re.search(r'^Fire Fighting Procedures:((.*|\n)+(?!\n\s{3,}))', text, re.M),
                              re.search(r'^Extinguishing Media:((.*|\n)+(?!\n\s{3,}))', text, re.M)]
        mapped_data = map(self.clean_text, procedures_results)
        procedures_data = dict(zip(PROCEDURES_DESCRIPTION_FEATURES, mapped_data))

        # description personal data
        personal_results = [re.search(r'^Respiratory Protection:((.*|\n)+(?!\n\s{3,}))', text, re.M),
                            re.search(r'^Other Protective Equipment:((.*|\n)+(?!\n\s{3,}))', text, re.M),
                            re.search(r'^Other Precautions:((.*|\n)+(?!\n\s{3,}))', text, re.M),
                            re.search(r'^Protective Gloves:((.*|\n)+(?!\n\s{3,}))', text, re.M),
                            re.search(r'^Eye Protection:((.*|\n)+(?!\n\s{3,}))', text, re.M),
                            re.search(r'^Ventilation:((.*|\n)+(?!\n\s{3,}))', text, re.M)]
        mapped_data = map(self.clean_text, personal_results)
        personal_data = dict(zip(PERSONAL_DESCRIPTION_FEATURES, mapped_data))

        # description chemical data
        chemical_results = [re.search(r'^Appearance and Odor:((.*|\n)+(?!\n\s{3,}))', text, re.M),
                            re.search(r'^Stability Condition to Avoid:((.*|\n)+(?!\n\s{3,}))', text, re.M),
                            re.search(r'^Hazardous Decomposition Products:((.*|\n)+(?!\n\s{3,}))', text, re.M)]
        mapped_data = map(self.clean_text, chemical_results)
        chemical_data = dict(zip(CHEMICAL_DESCRIPTION_FEATURES, mapped_data))

        # description health data
        # exp_carcinogenicity = clean_text(r'^Explanation of Carcinogenicity:((.*|\n)+(?!\n\s{3,}))')
        # health_data['extra_carcinogenicity'] = exp_carcinogenicity

        # listed chemical data
        # materials_avoid = re.search(r'^Stability Indicator\/Materials to Avoid:'
        #                             r'(.+)\n(([A-Z0-9]|[^A-Za-z0-9]){2,}\n)+(?!\n\s{3,})', text, re.M)
        # materials_avoid = materials_avoid.group(2) if materials_avoid is not None else None
        # ingredients = re.findall(r'^Ingred Name:((.*|\n)+(?!\n\s{3,}))', text, re.M)

        # ******************* OTHER DATA *******************

        # other data
        # evaporation_rate = re.search(r'^Evaporation Rate & Reference:((.*|\n)+(?!\n\s{3,}))', text, re.M)

        # carcinogenicity (BOOL DATA)
        carcinogenicity = re.search(r'^Reports of Carcinogenicity:NTP:(\w+)\s+IARC:(\w+)\s+OSHA:(\w+)', text, re.M)
        # convert to boolean data
        mapped_data = map(lambda x: self.clean_boolean(carcinogenicity, x), [1, 2, 3])
        bool_data = dict(zip(BOOL_FEATURES, mapped_data))

        # numerical data
        boiling_point = re.findall(r'^Boiling Pt:B\.P\.\s+Text:((.*)+(?!\n\s{3,}))', text, re.M)
        boiling_point, _ = boiling_point[0] if len(boiling_point) > 0 else (None, None)
        vapor_density = re.search(r'^Vapor Density:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        vapor_pressure = re.search(r'^Vapor Pres:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        flash_point = re.search(r'^Flash Point:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        # convert to numerical data
        mapped_data = map(self.clean_numerical, [boiling_point,
                                                 self.group_extraction(vapor_density),
                                                 self.group_extraction(vapor_pressure),
                                                 self.group_extraction(flash_point)])
        numerical_data = dict(zip(NUMERICAL_FEATURES, mapped_data))

        # zip the numerical data in a dict
        numerical_data = {**identification_data, **bool_data, **numerical_data}
        # return a dict with lists of keys extracted
        features_dict = {'numerical': numerical_data,
                         'health': health_data,
                         'personal': personal_data,
                         'chemical': chemical_data,
                         'procedures': procedures_data}
        return features_dict

    @staticmethod
    def group_extraction(search_result):
        if search_result is not None:
            # return the value if exist
            return search_result.group(1)
        else:
            # return None if there are no value
            return None

    @staticmethod
    def clean_boolean(bool_data, group_n):
        # checks if is none
        if bool_data is not None:
            # checks the string value
            if bool_data.group(group_n) == 'YES':
                return 1
            elif bool_data.group(group_n) == 'NO':
                return 0
        # if there is no value
        return None

    @staticmethod
    def clean_numerical(text_digits):
        if text_digits is not None:
            digits_list = re.findall(r'(\d+(?:\.\d+)?)', text_digits)
            digits = list(map(lambda x: float(x), digits_list))
            # returns the mean if len is 0 returns none
            if len(digits) > 0:
                return np.mean(digits)
        # if there is no value
        return None

    @staticmethod
    def clean_name(line):
        # check if is none
        if line is None:
            return None
        # removes the new lines
        str_clean = re.sub(r'(\n)', r'', line)
        # removes the slash
        str_clean = re.sub(r'(/)', r' ', str_clean)
        # removes the multiples spaces
        str_clean = re.sub(r'(\s+)', r' ', str_clean)
        # search not known values
        nk0 = re.search(r'((NONE).+(KNOWN|SPECIFIED))|(NOT.+KNOWN)|(^N\.A\.$)|(^NK$)|(^NON-RECOGNIZED$)',
                        str_clean, flags=re.IGNORECASE)
        # check the result and return
        return str_clean.lower() if nk0 is None else None

    # method to clean a description
    @staticmethod
    def clean_text(search_result):
        # if None result return dict with no strings
        if search_result is None:
            return None
        # otherwise stores the group
        text_clean = search_result.group(1)
        # removes the new lines
        text_clean = re.sub(r'(\n)', r'', text_clean)
        # removes the multiples spaces
        text_clean = re.sub(r'(\s+)', r' ', text_clean)
        # return the clean text
        return text_clean

    @staticmethod
    def clean_health_description(tag_name, search_result):
        tag2search = ['inhalation', 'ingestion', 'skin', 'eyes', 'chronic']
        composed_tags = list(map(lambda x_tag: '{0}_{1}'.format(tag_name, x_tag), tag2search))

        # if None result return dict with no strings
        if search_result is None:
            return dict(zip(composed_tags, [None] * 5))
        # otherwise stores the group
        text_clean = search_result.group(1)
        # removes the new lines
        text_clean = re.sub(r'(\n)', r'', text_clean)
        # removes the multiples spaces
        text_clean = re.sub(r'(\s+)', r' ', text_clean)

        re_exp = [r'^inhalation:(.+)|(inhalation.+)',
                  r'^ingestion:(.+)|(ingestion.+)',
                  r'^skin:(.+)|(skin.+)',
                  r'^eyes?:(.+)|(eyes?.+)',
                  r'^chroni\s?c:(.+)|(chroni\s?c.+)']

        # tokenizes the text
        sent_tokens = nltk.sent_tokenize(text_clean.lower())

        captured_sentences = []
        # iterate over each sentence
        for sentence in sent_tokens:
            # process each
            for re_query in re_exp:
                # search in the sentence
                search_r = re.search(re_query, sentence, re.M)
                # if finds anything
                if search_r is not None:
                    # otherwise process the sentence
                    sentence_r = search_r.group(1) if search_r.group(1) is not None else search_r.group(2)
                    captured_sentences.append(sentence_r)
                else:
                    captured_sentences.append('')
        # update the resulting dict
        return dict(zip(composed_tags, captured_sentences))
