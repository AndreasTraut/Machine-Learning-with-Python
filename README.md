Author: Andreas Traut    
Date: 08.05.2020  (Updates 24.07.2020)  
[Download as PDF](https://github.com/AndreasTraut/Machine-Learning-with-Python/raw/master/Machine-Learning-with-Python_AndreasTraut.pdf)

[TOC]


# Machine Learning with Python

## 0. Introduction

### a) Aim of this repository

After having learnt visualization techniques in Python (which I showed in my repository "Visualization-of-Data-with-Python", see https://github.com/AndreasTraut/Visualization-of-Data-with-Python) I started working on different datasets with the aim to apply machine learning algorithms (e.g. [Decision Tree](https://de.wikipedia.org/wiki/Entscheidungsbaum) to name only one). 

In my *first example* in this repository here I will use [Jupyter-Notebooks](https://jupyter.org/), which is a widespread standard today. The first Jupyter-Notebooks have been developed 5 years ago (in 2015). Since my first programming experience was more than 25 years ago (I started with [GW-Basic](https://de.wikipedia.org/wiki/GW-BASIC) then [Turbo-Pascal](https://de.wikipedia.org/wiki/Turbo_Pascal) and so on...) I quickly learnt the advantages of using Jupyter-Notebooks. 

**But** I missed the comfort of an [IDE](https://de.wikipedia.org/wiki/Integrierte_Entwicklungsumgebung) from the very first days!

Therefore: in my examples in this repository here I will also work with Python ".py" files. These ".py" can be executed in an IDE, like e.g. Spyder-IDE (see https://www.spyder-ide.org/)

![](./media/spyder.jpg)

### b) Motivation for IDEs

Why is it important for me to mention the IDEs out so early in a learning process? In my opinion Jupyter-Notebooks are good for the first examinations of data and for documenting procedures and up to a certain degree also for sophisticated data science. But it might be a good idea to learn very early how to work with an IDE. Think about how to use what has been developed so far later in a bigger environment (for example a [Lambda-Architecture](https://de.wikipedia.org/wiki/Apache_Hadoop#Lambda-Architektur), but you can take whatever other environment, which requires robustness&stability). I point this out here, because after having read several e-Books and having participated in seminars I see that IDEs are not in the focus.  

### c) Structure of this repository

#### (i) First part

Therefore the *first example* uses a [Jupyter-Notebook](https://jupyter.org/) in order to learn the standard procedures (e.g. data-cleaning & preparing, model-training,...). 

#### (ii) Second part ("small data")

The *second example* is for being used in an IDE (developer environment), like the [Spyder-IDE](https://www.spyder-ide.org/) from the [Anaconda distribution](https://www.anaconda.com/) and apply the ["Scikit-Learn Python Machine Learning Library"](https://scikit-learn.org/stable/)  (you may call this example a "small data" example if you want). I will show you a typical structure for a machine-learning example and put it into a mind-map. The same structure will be applied on the third example. 

#### (iii) Third part ("big data")

The *third example* is a "Big Data" example and will use a [Docker environment](https://www.docker.com/) and apply the ["Apache Machine Learning Library"](https://spark.apache.org/mllib/), a scalable machine learning library. The mind-map from the second part will be extended and aligned to the second example. In this example I also show some Big Data Visualizations techniques and also show how the K-Means Clustering Algorithm can be applied by using Apache Spark ML. 

#### (iv) Summary mind-map

I provide a summary mind-map, which possibly helps you to structure your code. Many steps in the "Skikit-Learn" example and the "Apache MLib" example are very similar and repeating. 

### d) Future learnings and coding & data sources 

For <u>all</u> of these topics various tutorials, documentation, coding examples and guidelines can be found in the internet **for free**! The Open Source Community is an incredible treasure trove and enrichment that positively drives many digital developments: [Scikit-Learn](https://github.com/scikit-learn/scikit-learn), [Apache Spark](https://github.com/apache/spark), [Spyder](https://github.com/spyder-ide/spyder), [GitHub](https://github.com/github), [Tensorflow](https://github.com/tensorflow/tensorflow) and also [Firefox](https://github.com/mozilla), [Signal](https://github.com/signalapp), [Threema](https://github.com/threema-ch), [Corona-Warnapp](https://github.com/corona-warn-app)... to be mentionned. There are many positive examples of sharing code and data "for free". 

<u>Coding:</u> 

If you Google for example *"how to prepare and clean the data with spark"*, you will find tons of documents around *"removing null values"* or *"encoders"* (like the OneHotEncoder for treating categorical inputs) or *"pipelines"* (for putting all the steps in an efficient, customizable order) so on. You will be overwhelmed of all this. Some resources to mention are the [official documentation](https://spark.apache.org/docs/latest/ml-guide.html) and a few more Github repositories like e.g. [tirthajyoti/Spark-with-Python](https://github.com/tirthajyoti/Spark-with-Python) (MIT licence), [Apress/learn-pyspark](https://github.com/Apress/learn-pyspark) (Freeware License), [mahmoudparsian/pyspark-tutorial](https://github.com/mahmoudparsian/pyspark-tutorial) (Apache License v2.0). What I will do here in my repository is nothing more than putting it together so that it works for my problem (which can be challenging as well sometimes). Adapting it for your needs should be easier from this point on. 

<u>Data:</u>

If you would like to do further analysis or produce alternate visualizations of the Airbnb-data, you can download them from [here](http://insideairbnb.com/get-the-data.html). It is available below under a [Creative Commons 1.0 Universal "Public Domain Dedication"](http://creativecommons.org/publicdomain/zero/1.0/) license. The data for the Vermont-Vendor-Payments can be downloaded from [here](https://data.vermont.gov/Finance/Vermont-Vendor-Payments/786x-sbp3) and are available under the [Open Data Commons Open Database License](http://opendatacommons.org/licenses/odbl/1.0/). The movies database doesn't even mention a license and is from [Kaggle](https://www.kaggle.com/isaactaylorofficial/imdb-10000-most-voted-feature-films-041118/metadata). There you find a lot of more datasets and also coding examples for your studies. 

## I. Movies Database Example

A good starting point for finding useful datasets is "Kaggle" (www.kaggle.com). I downloaded the movies dataset from [here](https://www.kaggle.com/isaactaylorofficial/imdb-10000-most-voted-feature-films-041118). The dataset from Kaggle contains the following columns:

Rank | Title | Year | Score | Metascore | Genre | Vote | Director | Runtime | **Revenue** | Description | RevCat

In this example I want to predict the **"Revenue"** based on the other information, which I have for each movie (e.g. every movie has a year, a scoring, a title ...). There are some "NaN"-values in the column "Revenue" and instead of filling them with an assumption (e.g. median-value) as I did in another Jupiter-Notebook (see [here](https://github.com/AndreasTraut/Machine-Learning-with-Python/blob/master/Movies%20Machine%20Learning%20-%20StratifiedSample.ipynb)), I wanted to predict these values. You might guess the conclusion already: predicting the revenue based on the available information as shown above (the columns) might not work. But essential to me is more to follow a well established standard-process of data-cleaning, data-preparing, model-training and error-calculation in this example in order to learn how to apply this process to better datasets, than the movies-dataset, later. 

Therefore, here is how I approached the problem step-by-step: 

#### Separate "NaN"-values

I separated the rows with "NaN"-values in column "Revenue"

![](./media/NaN_rows_in_Revenue.jpg)

#### Draw a stratified sample

I drew a stratified sample (based on "Revenue") on this remaining dataset and I received a training dataset and testing dataset:

![](./media/drew_stratified_sample.jpg)

![movies_train_test_nan](https://user-images.githubusercontent.com/55921277/79441450-87b98500-7fd7-11ea-80db-4630b1cbe123.png)

#### Create a pipeline

I created a pipeline to fill the "NaN"-value in other columns (e.g. "Metascore", "Score").

![](./media/create_pipeline.jpg)

![](./media/apply_pipeline.jpg)

#### Fit the model

I used the training dataset and fittet it with the "DecisionTreeRegressor" model

![](./media/fit_model_decisiontreeregresson.jpg)

#### Cross-validation

I verified with a cross-validation, how good this model/parameters are

![](./media/cross_validation.jpg)

#### Prediction 

I did a prediction on a subset of the testing dataset and did a side-by-side comparison of prediction and true value

![](./media/side_by_side_comparison.jpg)

I performed a prediction on the testing dataset and calculated the mean-squared error

![](./media/calculate_mean_squared_error.jpg)

#### Conclusion 

The conclusion of this machine learning example is obvious: it is rather not possible to predict the "Revenue" based on the available information (the most useful numerical features were "year", "score", ... and the other categorical like "genre" don't seem to have much more added value in my opinion). 

Please find the complete Jupyter Notebook here: 

https://github.com/AndreasTraut/Machine-Learning-with-Python/blob/master/Movies%20Machine%20Learning%20-%20Predict%20NaNs.ipynb

If you want to run the code immediately without installing the required "Jupyter environment" then you can use this Deepnote-Link:  

https://beta.deepnote.com/project/754094f0-3c01-4c29-b2f3-e07f507da460


## II. Step-by-step Python-Code for Machine Learning using "scikit-learn"

As said above, the *second example* is for being used in an IDE. If will use the ["Scikit-Learn"](https://scikit-learn.org/stable/) python machine learning library. As Jupyter Notebooks are not always the best environment for Python coding (e.g. Debugging), I extracted the most essential parts of Chapter 2 of Aurelien Geron "Machine Learning with Scikit-Learn & Tensorflow", sorted and arranged the code fragments slightly and created the following structured Python code for being used for example in the [Spyder-IDE](https://www.spyder-ide.org/). The structure of the Python code is a bit similar to the steps, which I followed in the Jupyter Notebooks above and are as follows (you will find these sections also in the ".py" file). Let's start with the "scikit-learn" ("SmallData", if you want): 

![](./media/MindMap_SkLearn.jpeg)

#### 1. create index   

##### 1.1 Alternative 1: generate id with static data

![](./media/1_1_generate_id_with_static_data.jpg)



##### 1.2 Alternative 2: generate stratified sampling

![](./media/1_2_generate_stratified_sampling.jpg)

##### 1.3 verify if stratified example is good

![](./media/1_3_verify_if_stratified_example_is_good.jpg)

#### 2. Discover and visualize the data to gain insights

![](./media/2_discover_and_visualize.jpg)

#### 3. prepare for Machine Learning  

##### 3.1 find all NULL-values

![](./media/3_1_find_all_NULL_values.jpg)

##### 3.2 remove all NULL-values

![](./media/3_2_remove_all_NULL_values.jpg)

#### 4. Use "Imputer" to clean NaNs

![](./media/4_use_imputer_to_clean_NaNs.jpg)

#### 5. treat "categorial" inputs

![](./media/5_treat_categorial_inputs.jpg)

#### 6. custom transformer and pipelines  

##### 6.1 custom transformer

![](./media/6_1_custom_transformer.jpg)

#####  6.2 pipelines

![](./media/6_2_pipelines.jpg)

#### 7. select and train model  

##### 7.1 LinearRegression model

![](./media/7_1_linear_regression_model.jpg)

#####  7.2 DecisionTreeRegressor model

![](./media/7_2_decisiontreeregressor_model.jpg)

#### 8. crossvalidation  

##### 8.1 for DecisionTreeRegressor

![](./media/8_1_crossvalidation_for_decisontreeregressor.jpg)

#####  8.2 for LinearRegression

![](./media/8_2_crossvalidation_for_linearregression.jpg)

##### 8.3 for RandomForestRegressor

![](./media/8_3_crossvalidation_for_randomforestregressor.jpg)

##### 8.4 for ExtraTreesRegressor

![](./media/8_4_crossvalidation_for_extratreesregressor.jpg)

#### 9. Save Model

![](./media/9_save_model.jpg)

#### 10. Optimize Model  

#####  10.1 GridSearchCV

###### 	 10.1.1 GridSearchCV on RandomForestRegressor	 

![](./media/10_1_1_gridsearchcv_randomforestregressor.jpg)

###### 	 10.1.2 GridSearchCV on LinearRegressor

![](./media/10_1_2_gridsearchcv_linearregressor.jpg)

#####  10.2 Randomized Search

![](./media/10_2_randomized_search.jpg)

#####  10.3 Analyze best models

![](./media/10_3_analyze_best_models.jpg)

#### 11. Evaluate final model on test dataset

![](./media/11_evaluate_final_model.jpg)


## III. Step-by-step Python-Code for Machine Learning using "Apache Machine-Learning" Libary (Big Data)

This will be an example for a ["Big-Data"](https://de.wikipedia.org/wiki/Big_Data) environment and uses the ["Apache MLib"](https://spark.apache.org/mllib/) scalable machine learning library. Various tutorials, documentation, "code-fragments" and guidelines can be found in the internet **for free** (at least for your private use). The best is in my opinion the [official documentation](https://spark.apache.org/docs/latest/ml-guide.html). A few more helpful sources are the following Github repositories: 

- [tirthajyoti/Spark-with-Python](https://github.com/tirthajyoti/Spark-with-Python) (MIT licence)
- [Apress/learn-pyspark](https://github.com/Apress/learn-pyspark) (Freeware License)
- [mahmoudparsian/pyspark-tutorial](https://github.com/mahmoudparsian/pyspark-tutorial) (Apache License v2.0)

Concerning the topic **"Big Data"** I want to add the following: I passed a certification as *"Data Scientist Specialized in Big Data Analytics"*. I must say: Understanding the concept of "Big-Data" and how to differenciate "standard" machine learning from a "scalable" environment is not easy. I recommend a separate training! Some steps are a bit similar to "scikit-learn" (e.g. data-cleaning, preprocessing), but the technical environment for running the code is different and also the code itself is different. 

Let's start with the structure, which I put into a mind map (you can download it from this repository). 

![](./media/MindMap_Spark.jpeg)

There are different ways to approach the Apache Spark and Hadoop environment: you can install it on your own computer (which I found rather difficult because of lack of userfriendly and easy understandable documentation). Or you can dive into a Cloud environment, like e.g. Microsoft Azure or Amazon EWS or Google Cloud and try to get a virtual machine up and running for your purposes. Have a look at my [documentation](https://github.com/AndreasTraut/Experiences-with-MicrosoftAzure), where I shared my experiences, which I had with Microsoft Azure [here](https://github.com/AndreasTraut/Experiences-with-MicrosoftAzure). 

For the following explanation I decided to use [Docker](https://www.docker.com/). What is Docker? Docker is *"an open-source project that automates the deployment of software applications inside containers by providing an additional layer of abstraction and automation of OS-level virtualization on Linux."* Learn from the [Docker-Curriculum](https://docker-curriculum.com/) how it works. I found an container, which had Apache Spark Version 3.0.0 and Hadoop 3.2 installed and built my machine-learning code (using pyspark) on top of this container. 

I shared my code and developments on Docker-Hub in the following repository [here](https://hub.docker.com/repository/docker/andreastraut/machine-learning-pyspark). After having installed the Docker application you will need to pull my "machine-learning-pyspark" image to your computer: 

`docker pull andreastraut/machine-learning-pyspark`

Then open Windows Powershell and type the following: 

`docker run -dp 8888:8888 andreastraut/machine-learning-pyspark:latest`

![](./media/docker_run.jpg)

You will see in your Docker Dashborad that a container is running: 

![](./media/docker_openbrowser.jpg)

After having opened your browser (e.g. Firefox-Browser), navigate to "localhost:8888" (8888 is the port, which will be opened). 

![](./media/docker_localhost.jpg)

The folder "data" contains the datasets. If you would like to do further analysis or produce alternate visualisations of the Airbnb-data, you can download them from [here](http://insideairbnb.com/get-the-data.html). It is available below under a [Creative Commons CC0 1.0 Universal (CC0 1.0) "Public Domain Dedication"](http://creativecommons.org/publicdomain/zero/1.0/) license. The data for the Vermont-Vendor-Payments can be downloaded from [here](https://data.vermont.gov/Finance/Vermont-Vendor-Payments/786x-sbp3) and are available under the [Open Data Commons Open Database License](http://opendatacommons.org/licenses/odbl/1.0/). 

![](./media/docker_localhost_data.jpg)

| Excurs to Big Data Visualization and K-Means Clustering Algorithm |
| ------------------------------------------------------------ |
| You will see a Jupyter-Notebook (which contains the Machine-Learning Code) and a folder named "data" (which contains the raw-data and preprocessed data). As you can see: I also worked on a 298 MB big csv-file (["Vermont_Vendor_Payments.csv"](https://data.vermont.gov/Finance/Vermont-Vendor-Payments/786x-sbp3)), which I coulnd't open in Excel, because of the huge size. This file contains a list of all state of Vermont payments to vendors (Open Data Commons Licence) and has  more than 1.6 million lines (exactly 1'648'466 lines). I already mentionned in my repository ["Visualization-of-Data-with-Python"](https://github.com/AndreasTraut/Visualization-of-Data-with-Python), that the **visualization of big datasets** can be difficult when using "standard" office tools, like Excel. If you are not able to open such csv-files in Excel you have to find other solutions. One is to use PySpark which I will show you here. Another solution would have been to use the Excel built-in connection, [PowerQuery](https://support.microsoft.com/de-de/office/einf%C3%BChrung-in-microsoft-power-query-f%C3%BCr-excel-6e92e2f4-2079-4e1f-bad5-89f6269cd605) or something similar, maybe Access or whatever, which is not the topic here, because we also want to be able to apply machine-learning algorithms from the  [Spark Machine Learning Library](https://spark.apache.org/mllib/). And there are more benefits of using PySpark instead of Excel: it can handle distributed processing, it's a lot faster, you can use pipelines, it can read many file systems (not only csv), it can process real-time data. <br /><br />Additionally I will work on this dataset to show how the **K-Means Clustering Algorithm** can be applied by using the Spark Marchine-Learning Libary (see more documentation [here](https://spark.apache.org/docs/latest/ml-clustering.html#k-means)). I will show how the "Vermont Vendor Payments" dataset can be clustered. Here every color represents a differents cluster: <br /><br />![](./media/k-means_clustering_DepAmount_NonScaled.png)![](./media/k-means_clustering_StateAmount_Scaled.png)![](./media/k-means_clustering_StateDep_NonScaled.png)<br /><br />The first plot shows 4 clusters, each in a different color. For example the violet cluster has values beween 0 and about 0.8 * 10e9 and 2.5 * 10e9. I will show you, how this can be calculated and visualized based on the 298 MB / 1.6 million lines of data. |



When you open the Jupyter-Notebook, you will see, that Apache Spark Version 3.0.0 and Hadoop Version 3.2 is installed:

![](./media/docker_jupyter_apache_spark.jpg)

Initializing a Spark sessions works and reading a CSV file can by done with the following commands (see more documentation [here](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.SparkSession) and also have a look at a ["Get Started Guide"](https://spark.apache.org/docs/latest/sql-getting-started.html#starting-point-sparksession)):

![](./media/docker_jupyter_initialize_spark.jpg)

After then the data-cleaning and data preparation (eliminating of null values, visualization techniques) work pretty similar to the "Small data" (Sklearn) approach. If you want to persist (=save) your intermediate you can do it as follows: 

![](./media/docker_jupyter_persisting_data.jpg)

I included some examples of how features can be extracted, transformed and selected in the Jupyter-Notebook (see more documentation [here](https://spark.apache.org/docs/latest/ml-features.html)). Just to mention a few here: the ["StringIndexer"](https://spark.apache.org/docs/latest/ml-features.html#stringindexer), ["OneHotEncoder"](https://spark.apache.org/docs/latest/ml-features.html#onehotencoder) and ["VectorAssembler"](https://spark.apache.org/docs/latest/ml-features.html#vectorassembler) work as follows: 

![](./media/docker_jupyter_stringindexer.jpg)

![](./media/docker_jupyter_onehotencoder_vectorassembler.jpg)

After having extracted, transformed and selected features you will want to apply some models, which are documented [here](https://spark.apache.org/docs/latest/ml-classification-regression.html), for example the ["OLS Regression](https://spark.apache.org/docs/latest/ml-classification-regression.html#linear-regression)": 

![](./media/docker_jupyter_ordinary_least_square_regression.jpg)

To summarize the whole coding structure have a look at the following and also the provided mind-maps: 

#### 0. Initialize Spark     

##### 0.1 Create Spark Context and Spark Session  

##### 0.2 Read CSV  

##### 0.3 Dataset Properties and some Select, Group and Aggregate Methods  

##### 0.4 Write as Parquet or CSV  

##### 0.5 Read Parquet  

##### 0.6 How to stop a Spark Session and Spark Context  

#### 1. Cleaning the data     

##### 1.1 Show number of rows and columns and do some visualizations  

##### 1.2 Replacing and Casting  

##### 1.3 Null-Values  

##### 1.4 String Values  

#### 2. Model-specific preprocessing    

##### 2.0 Check missing entries and define userdefined scatter plot  

##### 2.1 StringIndexer  

##### 2.2 OntHotEncoder  

##### 2.3 VectorAssembler  

##### 2.4 CountVectorizer  

#### 3. Aligning and numerating Features and Labels    

##### 3.1 Aligning  

##### 3.2 Numerating  

#### 4. Pipelines  

#### 5. Training data and Testing data  

#### 6. Apply models and evaluate    

##### 6.1 Ordinary Least Square Regression  

##### 6.2 Ridge Regression  

##### 6.3 Lasso Regression  

##### 6.4 Decision Tree  

#### 7. Minhash und Local-Sensitive-Hashing (LSH) 

see example: https://github.com/AndreasTraut/Deep_learning_explorations

#### 8. Alternative-Least-Square (ALS)    

##### 8.1. Datapreparation for ALS  

##### 8.2 Build the recommendation model using alternating least squares (ALS)  

##### 8.3 Get recommendations  

##### 8.4 Clustering of Users with K-Means 

see example: https://hub.docker.com/repository/docker/andreastraut/machine-learning-pyspark  

##### 8.5 Perform a PCA and draw the 2-dim projection  



## IV. Summary Mind-Map

My mind map below may help you to structure your code: 

![](./media/MindMap_SkLearn_and_Spark.jpeg)

---

# MIT License

Copyright (c) 2020 Andras Traut

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.