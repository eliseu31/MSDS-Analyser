# Datasheet Analyser

Dataset size 253963 files, 1.5GB 
(f1 -> 17454 files, 155.4MB) 
(f2 -> 236507 files, 1.3GB)

The main goals are:
* Check the relations between the manufacturers, ingredients and the disposal methods

Questions:
* Process description text to extract features?
* Classifier output must be value or description? (clustering in the disposal)
* Witch variables to use?


- [x] Data scrapping from the hazard.com site
- [x] Read the files and split the files categories using regex
- [ ] Clean features
- [ ] Process text description categories and extract features

## Papers

https://www.sciencedirect.com/science/article/abs/pii/S0091674902000374
https://www.sciencedirect.com/science/article/abs/pii/S019606449670272X


## Data Structure

There are 2 different structures of file F1 and F2.

| Features                 |  Category  |       Type       | Model  |
|:-------------------------|:----------:|:----------------:|:------:|
| file_name                |    info    |    identifier    |   -    |
| product_name             |    info    |  value: string   |   -    |
| company                  |    info    |  value: string   | Input  |
| appearance               |  chemical  | text description | Input  |
| boiling_point            |  chemical  |   value: range   | Input  |
| vapor_density            |  chemical  |   value: range   |        |
| vapor_pressure           |  chemical  |                  |        |
| ingredients              |  chemical  |       list       | Input  |
| evaporation_rate         |  chemical  |  value: string   |        |
| flash_point              |  chemical  |   value: range   |        |
| decomposition_products   |  chemical  |       list       |        |
| conditions_avoid         |  chemical  |       list       |        |
| materials_avoid          |  chemical  |       list       |        |
| carcinogenicity_iarc     |   health   |    value: 1/0    | Input  |
| carcinogenicity_ntp      |   health   |    value: 1/0    | Input  |
| carcinogenicity_osha     |   health   |    value: 1/0    | Input  |
| extra_carcinogenicity    |   health   |                  |        |
| effects_overexposure     |   health   |       list       |        |
| health_hazards           |   health   |                  |        |
| eyes_protection          |  personal  |                  |        |
| protective_gloves        |  personal  |  value: string   |        |
| respiratory_protection   |  personal  | text description |        |
| ventilation              |  personal  |                  |        |
| other_equipment          |  personal  |                  |        |
| extinguishing_media      | procedures |       list       | Input  |
| fire_fighting_procedures | procedures | text description |        |
| first_aids               | procedures | list: (eyes,...) |        |
| other_precautions        | procedures |                  |        |
| spill_procedures         | procedures | text description |        |
| storage_precautions      | procedures | text description |        |
| disposal                 | procedures | text description | Target |

### Text Descriptions
Use n-grams, bag-of-words.
