# Counting Carbon: A Survey of Factors Influencing the Emissions of Machine Learning

This repository contains the data and code used in the paper [Counting Carbon: A Survey of Factors Influencing the Emissions of Machine Learning](https://arxiv.org/abs/2302.08476) (Luccioni and Hernandez-Garcia, 2023).

## Abstract

Machine learning (ML) requires using energy to carry out computations during the model training process. The generation of this energy comes with an environmental cost in terms of greenhouse gas emissions, depending on quantity used and the energy source. Existing research on the environmental impacts of ML has been limited to analyses covering a small number of models and does not adequately represent the diversity of ML models and tasks. In the current study, we present a survey of the carbon emissions of 95 ML models across time and different tasks in natural language processing and computer vision. We analyze them in terms of the energy sources used, the amount of CO2 emissions produced, how these emissions evolve across time and how they relate to model performance. We conclude with a discussion regarding the carbon footprint of our field and propose the creation of a centralized repository for reporting and tracking these emissions.

## Data

The main data of the paper is in contained in the CSV file [data/co2ml96models.csv](./data/co2ml96models.csv). Each row in the file contains information about one machine learning model, such as the carbon intensity of the main energy source, the training time, number of GPUs/TPUs, the estimated carbon emissions associated to training it, the performance metrics, etc. We have removed all information that would allow to directly identify the paper from the data base and we have also added a small amount of noise to some of variables, such as performance metrics.

## How to run the code

First of all, clone the repository into your local machine:

```bash
git clone git@github.com:alexhernandezgarcia/co2ml.git
cd co2ml
```

### Python virtual environment

You may want to setup a Python virtual environment before running the code:

```bash
python -m virtualenv env-co2ml
source env-co2ml/bin/activate
```

### Install dependencies

Install the Python dependencies explicitly:

```bash
python -m pip install numpy pandas matplotlib seaborn pyyaml scipy
```

OR via the requirements file:

```bash
python -m pip install -r requirements.txt
```

Now you should be ready to run the scripts!

### Main statistics script

You can run the script that retrieves and prints general statistics from the data with the following Python command:

```bash
python explore_data.py --data_csv data/co2ml96models.csv
```

### Figures

The scripts that generate the figures admit arguments that will change the appearance and properties of the plots. These arguments can be passed via the command line or via a YAML configuration file. In order to reproduce the figures from the paper and others, run the following commands that use configuration files contained in the repository:

```bash
python figures/gpuhours_task_year.py --input_csv data/co2ml96models.csv --y figures/config/gpuhours_task_year.yml
python figures/carbonintensity_sources_year.py --input_csv data/co2ml96models.csv --y figures/config/carbonintensity_sources_year.yml
python figures/co2_performance.py --input_csv data/co2ml96models.csv --y figures/config/co2_performance.yml
python figures/co2_sources_year.py --input_csv data/co2ml96models.csv --y figures/config/co2_sources_year.yml
python figures/co2_task_year.py --input_csv data/co2ml96models.csv --y figures/config/co2_task_year.yml
python figures/energy_carbonintensity_sources.py --input_csv data/co2ml96models.csv --y figures/config/energy_carbonintensity_sources.yml
python figures/energy_co2_sources.py --input_csv data/co2ml96models.csv --y figures/config/energy_co2_sources.yml
python figures/energy_performance.py --input_csv data/co2ml96models.csv --y figures/config/energy_performance.yml
python figures/energy_task_year.py --input_csv data/co2ml96models.csv --y figures/config/energy_task_year.yml
python figures/gpuhours_task_year.py --input_csv data/co2ml96models.csv --y figures/config/gpuhours_task_year.yml
python figures/map_carbonintensity.py --input_csv data/co2ml96models.csv --y figures/config/map_carbonintensity.yml
python figures/map_count_countries.py --input_csv data/co2ml96models.csv --y figures/config/map_count_countries.yml
python figures/sources_count_year.py --input_csv data/co2ml96models.csv --y figures/config/sources_count_year.yml
python figures/us_grid.py --input_csv data/usgrid.csv --y figures/config/us_grid.yml
```

## Citation

If you use this data or code for scientific purposes, please consider citing [Counting Carbon: A Survey of Factors Influencing the Emissions of Machine Learning](https://arxiv.org/abs/2302.08476):

	@article{luccioni2023co2ml,
    title={Counting Carbon: A Survey of Factors Influencing the Emissions of Machine Learning},
      author={Alexandra Sasha Luccioni and Alex Hernandez-Garcia},
      year={2023},
      journal={arXiv preprint arXiv: Arxiv-2302.08476}
    }

## License

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/)

[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
