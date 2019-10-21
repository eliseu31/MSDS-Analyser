# Datasheet Analyser

- [x] Data scrapping from the **hazard.com** site
- [x] Read the files and split the files categories using regex
- [x] Join all the categories in a single dataframe
- [ ] Use n-grams, pos_tag to process text description categories and extract features
- [ ] Apply different classification models

## Papers

* https://www.sciencedirect.com/science/article/abs/pii/S0091674902000374
* https://www.sciencedirect.com/science/article/abs/pii/S019606449670272X

## Data

Dataset size **253963** files, **1.5GB**: 
* **f1**: **17454** files, **155.4MB** 
* **f2**: **236507** files, **1.3GB**


### Data Structure

| Features                 |  Category  |       Type       | Model  |
|:-------------------------|:----------:|:----------------:|:------:|
| file_name                |    info    |    identifier    |   -    |
| product_name             |    info    |  value: string   |   -    |
| company                  |    info    |  value: string   | Input  |
| appearance               |  chemical  | text description | Input  |
| boiling_point            |  chemical  |   value: range   | Input  |
| vapor_density            |  chemical  |   value: range   | Input  |
| vapor_pressure           |  chemical  |   value: range   | Input  |
| ingredients              |  chemical  |       list       | Input  |
| evaporation_rate         |  chemical  |  value: string   | Input  |
| flash_point              |  chemical  |   value: range   | Input  |
| decomposition_products   |  chemical  |       list       | Input  |
| conditions_avoid         |  chemical  |       list       | Input  |
| materials_avoid          |  chemical  |       list       | Input  |
| carcinogenicity_iarc     |   health   |    value: 1/0    | Input  |
| carcinogenicity_ntp      |   health   |    value: 1/0    | Input  |
| carcinogenicity_osha     |   health   |    value: 1/0    | Input  |
| extra_carcinogenicity    |   health   | text description | Input  |
| effects_overexposure     |   health   |       list       | Input  |
| health_hazards           |   health   | text description | Input  |
| eyes_protection          |  personal  |  value: string   | Input  |
| protective_gloves        |  personal  |  value: string   | Input  |
| respiratory_protection   |  personal  | text description | Input  |
| ventilation              |  personal  |  value: string   | Input  |
| other_equipment          |  personal  | text description | Input  |
| other_precautions        |  personal  | text description | Input  |
| extinguishing_media      | procedures |       list       | Target |
| fire_fighting_procedures | procedures | text description | Target |
| first_aids               | procedures | list: (eyes,...) | Target |
| spill_procedures         | procedures | text description | Target |
| storage_precautions      | procedures | text description | Target |
| disposal                 | procedures | text description | Target |

### Text Processing

* n-grams
* pos-tag
* bag-of-words

