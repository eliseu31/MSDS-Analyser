from joblib import Parallel, delayed
import pandas as pd
import re
import os
import sys


class TextExtractor:

    def __init__(self, data_folder='datasheets', n_jobs=8):
        # folder to save the dataset
        self.root_folder = os.path.join(os.path.dirname(sys.path[0]), data_folder)
        self.N_JOBS = n_jobs

    def __call__(self, *args, **kwargs):
        folders = {'f2': self.process_f2}
        # uses parallel processes
        with Parallel(n_jobs=self.N_JOBS) as parallel:
            # iterates over each parent folder
            for folder_name, folder_method in folders.items():
                folder_path = os.path.join(self.root_folder, folder_name)
                child_folders = os.listdir(folder_path)
                # split in N equal slices
                division = len(child_folders) // self.N_JOBS
                slices = [child_folders[division * i: division * (i + 1)] for i in range(self.N_JOBS)]
                slices[self.N_JOBS - 1] = child_folders[division * (self.N_JOBS - 1):]
                print(folder_name, 'size:    ', len(child_folders))
                print('slices size:', [len(slice_size) for slice_size in slices])
                # process the files
                partial_dfs = parallel(delayed(self.process_files)(folder_name,
                                                                   folder_method,
                                                                   partial_slice) for partial_slice in slices)
                # zip all the results
                material_df = pd.concat(partial_dfs, sort=True)
                material_df.set_index('file_name', inplace=True)
                # save the dataset
                print(material_df.info())
                material_df.to_csv('material.csv')
                return material_df

    def process_files(self, folder_name, method, partial_slice):
        materials_df = None
        # iterate over each folder
        for folder in partial_slice:
            folder_path = os.path.join(self.root_folder, folder_name, folder)
            # list all the file inside the folder
            file_names = os.listdir(folder_path)
            # iterate over each file
            for file_name in file_names:
                file_path = os.path.join(folder_path, file_name)
                # print(file_path)
                with open(file_path) as f:
                    text = f.read()
                    material_data = method(text)
                    material_data['file_name'] = file_name.split('.')[0]
                    # removes the ingredients
                    ingredients_data = material_data.pop('ingredients')
                    # build the ingredients df
                    if materials_df is None:
                        # create the materials df
                        materials_df = pd.DataFrame(columns=material_data.keys())
                    # append a row to the materials df
                    materials_df = materials_df.append(material_data, ignore_index=True)
            print('processed', partial_slice.index(folder), 'of', len(partial_slice))

        # resets the index
        return materials_df

    def process_f2(self, text):
        product_name = re.search(r'^Product ID:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        # pair responsible and contractor
        company_info = re.findall(r'^=+\s+(.*)\s+=+\nCompany Name:(.+)$', text, re.M)
        ingredients = re.findall(r'^Ingred Name:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        # text to split after
        first_aids = re.search(r'^First Aid:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        # accidental measures
        spill_procedures = re.search(r'^Spill Release Procedures:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        # neutralizing_agent = re.search(r'^Neutralizing Agent:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        # handling and storage
        storage_precautions = re.search(r'^Handling and Storage Precautions:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        other_precautions = re.search(r'^Other Precautions:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        # hazards
        effects_overexposure = re.search(r'^Effects of Overexposure:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        health_hazards = re.search(r'^Health Hazards Acute and Chronic:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        carcinogenicity = re.search(r'^Reports of Carcinogenicity:NTP:(\w+)\s+IARC:(\w+)\s+OSHA:(\w+)', text, re.M)
        extra_carcinogenicity = re.search(r'^Explanation of Carcinogenicity:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        # fire fighting
        flash_point = re.search(r'^Flash Point:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        extinguishing_media = re.search(r'^Extinguishing Media:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        fire_fighting_procedures = re.search(r'^Fire Fighting Procedures:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        # personal protections
        respiratory_protection = re.search(r'^Respiratory Protection:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        ventilation = re.search(r'^Ventilation:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        protective_gloves = re.search(r'^Protective Gloves:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        eyes_protection = re.search(r'^Eye Protection:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        other_equipment = re.search(r'^Other Protective Equipment:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        # physical/chemical properties
        boiling_point = re.findall(r'^Boiling Pt:B\.P\.\s+Text:((.*)+(?!\n\s{3,}))', text, re.M)
        vapor_density = re.search(r'^Vapor Density:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        vapor_pressure = re.search(r'^Vapor Pres:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        appearance = re.search(r'^Appearance and Odor:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        evaporation_rate = re.search(r'^Evaporation Rate & Reference:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        # stability and reactivity
        conditions_avoid = re.search(r'^Stability Condition to Avoid:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        decomposition_products = re.search(r'^Hazardous Decomposition Products:((.*|\n)+(?!\n\s{3,}))', text, re.M)
        materials_avoid = re.search(r'^Stability Indicator\/Materials to Avoid:'
                                    r'(.+)\n(([A-Z0-9]|[^A-Za-z0-9]){2,}\n)+(?!\n\s{3,})',
                                    text, re.M)
        # disposal
        disposal = re.search(r'^Waste Disposal Methods:((.*|\n)+(?!\n\s{3,}))', text, re.M)

        # string to remove new lines and spaces
        material_data = {'product_name': product_name,
                         'first_aids': first_aids,
                         'spill_procedures': spill_procedures,
                         'storage_precautions': storage_precautions,
                         'other_precautions': other_precautions,
                         'effects_overexposure': effects_overexposure,
                         'health_hazards': health_hazards,
                         'extra_carcinogenicity': extra_carcinogenicity,
                         'flash_point': flash_point,
                         'extinguishing_media': extinguishing_media,
                         'fire_fighting_procedures': fire_fighting_procedures,
                         'respiratory_protection': respiratory_protection,
                         'ventilation': ventilation,
                         'protective_gloves': protective_gloves,
                         'eyes_protection': eyes_protection,
                         'other_equipment': other_equipment,
                         'vapor_density': vapor_density,
                         'vapor_pressure': vapor_pressure,
                         'appearance': appearance,
                         'evaporation_rate': evaporation_rate,
                         'conditions_avoid': conditions_avoid,
                         'decomposition_products': decomposition_products,
                         'disposal': disposal
                         }
        # iterate to remove the new lines and the spaces
        for str_name, str_item in material_data.items():
            # get the string value
            str_value = str_item.group(1) if str_item is not None else None
            # stores the value in the dict
            material_data[str_name] = self.clean_string(str_value)

        # extract companies names
        _, material_data['company'] = company_info[0] if len(company_info) > 0 else (None, None)
        # carcinogenicity
        material_data['carcinogenicity_ntp'] = carcinogenicity.group(1) if carcinogenicity is not None else None
        material_data['carcinogenicity_iarc'] = carcinogenicity.group(2) if carcinogenicity is not None else None
        material_data['carcinogenicity_osha'] = carcinogenicity.group(3) if carcinogenicity is not None else None
        # boiling point
        material_data['boiling_point'], _ = boiling_point[0] if len(boiling_point) > 0 else (None, None)

        # extract the ingredients
        ingredients_list = []
        for ingredient, _ in ingredients:
            # stores the value in the dict
            ingredients_list.append(self.clean_string(ingredient))
        # save the ingredients
        material_data['ingredients'] = ingredients_list

        # get the material avoid value
        avoid_value = materials_avoid.group(2) if materials_avoid is not None else None
        # materials to avoid
        material_data['materials_avoid'] = self.clean_string(avoid_value)

        # variables with text to process

        return material_data

    @staticmethod
    def clean_string(line):
        # check if is none
        if line is None:
            return None
        # removes the new lines
        str_clean = re.sub(r'(\n)', r'', line)
        # removes the multiples spaces
        str_clean = re.sub(r'(\s+)', r' ', str_clean)
        # search not known values
        nk0 = re.search(r'((NONE).+(KNOWN|SPECIFIED))|(NOT.+KNOWN)|(^N\.A\.$)|(^NK$)|(^NON-RECOGNIZED$)',
                        str_clean, flags=re.IGNORECASE)
        # check the result and return
        return str_clean if nk0 is None else None
