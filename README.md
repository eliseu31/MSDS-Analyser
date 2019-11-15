# Datasheet Analyser

- [x] Data scrapping from the **hazard.com** site
- [x] Read the files and split the files categories using regex
- [x] Join all the categories in a single dataframe
- [ ] Use n-grams, pos_tag to process text description categories and extract features
- [ ] Use clustering to get the labels/classes
- [ ] Graphics to check some metrics
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
| product_name             |    info    |      string      |   -    |
| company                  |    info    |      string      | Input  |
| boiling_point            |  chemical  |     numerical    | Input  |
| vapor_density            |  chemical  |     numerical    | Input  |
| vapor_pressure           |  chemical  |     numerical    | Input  |
| flash_point              |  chemical  |     numerical    | Input  |
| evaporation_rate         |  chemical  |      string      | Input  |
| appearance               |  chemical  | text description | Input  |
| ingredients              |  chemical  |       list       | Input  |
| decomposition_products   |  chemical  |       list       | Input  |
| conditions_avoid         |  chemical  |       list       | Input  |
| materials_avoid          |  chemical  |       list       | Input  |
| effects_overexposure     |   health   |health description| Input  |
| carcinogenicity_iarc     |   health   |       1/0        | Input  |
| carcinogenicity_ntp      |   health   |       1/0        | Input  |
| carcinogenicity_osha     |   health   |       1/0        | Input  |
| extra_carcinogenicity    |   health   | text description | Input  |
| health_hazards           |   health   |health description| Input  |
| first_aids               |   health   |health description| Input  |
| eyes_protection          |  personal  |      string      | Input  |
| protective_gloves        |  personal  |      string      | Input  |
| ventilation              |  personal  |      string      | Input  |
| respiratory_protection   |  personal  | text description | Input  |
| protective_equipment     |  personal  | text description | Input  |
| other_precautions        |  personal  | text description | Input  |
| extinguishing_media      | procedures |       list       | Target |
| fire_fighting_procedures | procedures | text description | Target |
| spill_procedures         | procedures | text description | Target |
| storage_precautions      | procedures | text description | Target |
| disposal                 | procedures | text description | Target |

### Text Processing

* n-grams
* pos-tag
* bag-of-words

