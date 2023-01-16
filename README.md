
# Capstone-Project---Consumer-Complaints
 ![image](https://user-images.githubusercontent.com/110474324/211333748-4feec551-0311-4e89-a9b4-ac9c157b431b.png)
## Overview
#### Our study  will assist financial institutions  in identifying the types of complaints for resolution, leading to increased customer satisfaction to drive revenue and profitability in a multi-label multi-class classification machine learning project. The data for this project was sourced from
[link](https://catalog.data.gov/dataset/consumer-complaint-database)
## General Objective
To predict the likelihood of consumers disputing complaints responses made by financial service providers regarding products and services.
## Project Success Criteria
Correctly classify complaints to targets (company_response_to_consumer, consumer_disputed, timely_response, product) with a recall score of above 75%

## Features
#### The data set includes fields that represent consumer complaints across different banks and financial institutions The dataset contains 3147570 rows and 18 columns with no duplicates. The features include:
* Date received
* Product
* Sub-product 
* Issue
* Sub-issue 
* Consumer complaint narrative
* Company public response 
* Company
* State
* Zip code 
* Submitted via
* Tags
* Consumer consent provided
## Data Preparation
## The following steps were followed in preparing the data;
#### Checking for validity
#### Checking for consistency of data
#### Checking for data uniformity
#### Checking for completeness of the data
## Explatory Data Analysis
### Univariate Analysis
#### We shall analyze individual columns/variables to see how it’s distributed and get patterns from the column.
[link](https://docs.google.com/document/d/1ywXDHODoqUpEeNct7-XyqQsS9729d5PYkhZMwfas_is/edit#)
### Bivariate Analysis
####  Here we will analyze two columns against each other with the aim of getting  more information and establishing patterns
[link](https://docs.google.com/document/d/1ywXDHODoqUpEeNct7-XyqQsS9729d5PYkhZMwfas_is/edit#)
## Modelling
In this section, we built classification models using Bidirectional Encoder Representations from Transformers.
The technologies used include:
* Bert-base-uncased
* Albert-base-v2
### Model Evaluation
The models were evaluated using the metrics from the training logs that are stored in the tensorboards below:
* [tuned Bert-base-uncased](https://tensorboard.dev/experiment/mJDC3OFvS6mc5eafVhDEDg/#scalars) 
* [tuned albert-base-v2](https://tensorboard.dev/experiment/LQdz2432QDSMyidy6HAdRQ/#scalars)
### Authors and Aknowledgement:

Special thanks to our Moringa School Data science Techincal Mentors for their guidance throughout the process of our project and the abnormal distribute team members :point_down:

* [Teofilo Gafna](https://github.com/teofizzy)
* [Daniel Kimutai](https://github.com/danielkimutai)
* [Majorie Opiyo](https://github.com/Opiyow)
* [chris_kinyanjui](https://github.com/K1nyash)




