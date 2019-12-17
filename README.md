# Material Safety Data Sheet (MSDS) Analyser

- [x] Data scrapping from the **hazard.com** site
- [x] Read the files and split the files categories using regex
- [x] Join all the categories in a single dataframe
- [ ] Use n-grams, pos_tag to process text description categories and extract features
- [ ] Graphics to check some metrics
- [ ] Apply different classification models

## Data Organization

The full dataset contains **253963** files (datasheets), with a total size of **1.5GB**.
There are 2 different types of datasheet: 
the **f1** type that contains **17454** files (**155.4MB**) and 
the **f2** type that contains **236507** files (**1.3GB**).

The following table shows all the variables present in each file. 
The variables are grouped in categories: **chemical**, **health**, **personal** and **procedures**. 


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

## Text Processing


## Useful Papers

* https://www.sciencedirect.com/science/article/abs/pii/S0091674902000374
* https://www.sciencedirect.com/science/article/abs/pii/S019606449670272X
