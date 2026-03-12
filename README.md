# Street-Level Image Embeddings & Deprivation Analysis

This repository contains code for testing whether area-level deprivation can be predicted from Google Street View imagery using image embeddings.

This is part of the [INTEGRATE](https://urban-analytics.github.io/INTEGRATE/) project and originally lived in the
[INTEGRATE repository](https://github.com/Urban-Analytics/INTEGRATE), specfically 
[here](https://github.com/Urban-Analytics/INTEGRATE/tree/main/llm/python/Embeddings/CLIP/WithClustering) 
but we moved it to this repository to make it easier to 
find and use, and to consolidate the locations of the code and data (which were previously spread all over the place). 
Claude did the big refactor.

The workflow combines large-scale image sampling, CLIP-based feature extraction, spatial aggregation, and supervised 
modelling to examine relationships between visual characteristics of neighbourhoods and the UK Index of Multiple Deprivation (IMD).

## Overview

This repository provides an end-to-end workflow for:

- Sampling points across a road network and downloading Google Street View imagery (actually this is partly done by a separate script, but the output of that script is used as the input to this workflow)
- Generating CLIP image embeddings  
- Aggregating embeddings at the neighbourhood (LSOA) level  (median pooling)
- Fitting a predictive model for deprivation (IMD) using the median pooled embedding in each LSOA
- Exploring relationships between visual indicators and the Index of Multiple Deprivation (IMD)  


## Data Sources
### Street View Imagery
- Google Street View Static API
- Four images captured per location (one per cardinal direction)

### Deprivation Data
- Index of Multiple Deprivation (IMD), LSOA-level


## Methodological Summary

- **Sampling**: Points are sampled along the road network at approximately 200 m spacing.
- **Image Collection:** At each sampled point, four Street View images are downloaded facing north, south, east, and west.
- **Feature Extraction:** Each image is encoded using the CLIP image encoder, producing a 512-dimensional embedding.
- **Spatial Aggregation:** Image embeddings are aggregated to the LSOA level using median pooling.
- **Clustering**: Embeddings are clustered to test whether visually distinct image groupings exist.
- **Modelling**: Supervised models are trained to predict IMD using aggregated embedding features.

## Repository Structure

The workflow is designed to be run sequentially.

----------------
### 1-SampleStreetNetwork.ipynb  (_not needed to run_)

This actually combines raw street view images and their metadata, originally downloaded by [this script](https://github.com/Urban-Analytics/INTEGRATE/blob/main/llm/python/5-street_view_embeddings.ipynb)
and puts them into a H5 database for the later scripts.

**Output**: H5 database containing the raw images and their metadata.

----------------

### 2-CalculateEmbeddings.ipynb  

This notebook performs all embedding calculations using CLIP


#### Image Embeddings  

- Loads each Street View image and extracts its 512 character CLIP image embedding  
- Stores per-image:  
     - 512 character embedding  

NB: Also possible to calcualte similarity scores with CLIP to text embeddings. 
We trialled this in previous iterations of the work. 
Code for this is: https://github.com/Urban-Analytics/INTEGRATE/tree/main/llm/python/Embeddings/CLIP/WithSemanticLabels

**Output**: Dataset with one embedding vector per image.

----------------

### 3-ProcessEmbeddings+FindMedianEmbeddingPerLSOA.ipynb

Aggregates image-level outputs to the LSOA level.

This notebook:

- Computes the percentage of images in each category per LSOA  
- Generates mean, median, and max CLIP embeddings per LSOA  
- Repeats these calculations using only images belonging to each category  
- Produces a final LSOA-level dataframe containing:  
    - The overall mean/max/median LSOA embedding
    - Category proportions
    - The mean/max/median embedding for each of the categories

**Output**: Dataframe containing the overall LSOA mean / median / max embedding; 
Percentage of images in each cluster; Mean / median / max embedding per cluster

----------------

### 4-RunModelsWithMedianEmbedding.ipynb

Performs model selection, hyper-parameter tuning and out-of-sample model testing

This script:

- Reads in the dataframe containing the per-LSOA summary info (e.g. mean/max/min embedding, mean/max/min embedding per cluster grouping, % of images in each cluster in each LSOA)
- Splits the data into 80% training, 20% testing
- Performs model selection and hyper-parameter tuning using the 80% training data:
- Tests out-of-sample performance of ‘best’ model

**Output**: A saved version of the best model; statistics on out-of-sample model performance 

----------------

### 5-IdentifyOptimalClusterNumber.ipynb

Does various tests to check for the presence of structure within the embeddings that would allow them to be meaningfully broken down into sub-clusters

clustering_functions.py provides various functions used by the main jupyter notebook.

**Output**: Figures for use in the paper justifying how many clusters to use, a (subjective) answer to use 7 clusters going forward, Figures plotting examples images from each of the 7 clusters (or for different values of k, by adjusting this parameter at the start of the script) 

----------------

### 6-TestModelOverClusters_ControlledForSampleSize.ipynb

This script evaluates how predictive performance varies with sample size and number of clusters, while controlling for unequal image counts across clusters.

**Output**: A pickle file contaninig the results from these experiments. A figure plotting the results of these experiments (this can be created for various different error metrics by changing this parameter in the script). 

----------------

### 7-RunModels_ForEachOfNClusters.ipynb

This script computes per-LSOA embedding summary statistics (mean/median/max) within each cluster, then trains and evaluates a separate XGBoost model for each cluster using an 80/20 train-test split.

**Output**: A comparison of model performance (R², RMSE, Spearman correlation) across clusters, with scatter plots.

----------------

### 8-Prediction_Deprivation_domains.ipynb

Extends the per-cluster analysis to individual IMD deprivation domains (Income, Employment, Education, Health, Crime, Housing, Living Environment). For each domain, trains a separate XGBoost model per cluster and a global model using the overall median embedding.

The default display metric is **Normalised RMSE** (NRMSE = RMSE / std(y_test)), which is comparable across clusters despite each cluster having a different subset of LSOAs. R² and RMSE are also available via a configuration option.

**Output**: A results matrix with rows as IMD domains and columns as clusters + global, displayed as a heatmap.

----------------
