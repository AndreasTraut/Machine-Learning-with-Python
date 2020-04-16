# Machine Learning with Python
After having learnt some visualization techniques (which I showed in my repository "Visualization-of-Data-with-Python", https://github.com/AndreasTraut/Visualization-of-Data-with-Python) I started working on different datasets with the aim to apply machine learning techniques. In this example I will show some machine learning examples.

## 1. Movies Database Example

A good starting point for finding useful datasets is "Kaggle" (www.kaggle.com). I downloaded the following movies dataset: 

https://www.kaggle.com/isaactaylorofficial/imdb-10000-most-voted-feature-films-041118

The dataset from Kaggle contains the following columns:

Rank | Title | Year | Score | Metascore | Genre | Vote | Director | Runtime | **Revenue** | Description | RevCat

In this example I want to predict the **"Revenue"** based on the other information, which I have for each movie (e.g. every movie has a year, a scoring, a title ...). There are some "NaN"-values in the column "Revenue" and instead of filling them with an assumption (e.g. median-value) as I did in another Jupiter-Notebook (see here https://github.com/AndreasTraut/Machine-Learning-with-Python/blob/master/Movies%20Machine%20Learning%20-%20StratifiedSample.ipynb), I wanted to predict these values. 

Therefore I did the following:
- I separated the rows with "NaN"-values in column "Revenue"
- I drew a stratified sample (based on "Revenue") on this remaining dataset and I received a training dataset and testing dataset:

![movies_train_test_nan](https://user-images.githubusercontent.com/55921277/79441450-87b98500-7fd7-11ea-80db-4630b1cbe123.png)

- I created a pipeline to fill the "NaN"-value in other columns (e.g. "Metascore", "Score").
- used the training dataset and fittet it with the "DecisionTreeRegressor" model
- verified with a cross-validation, how good this model/parameters are
- did a prediction on a subset of the testing dataset and did a side-by-side comparison of prediction and true value
- performed a prediction on the testing dataset and calculated the mean-squared error

Please find the complete Jupyter Notebook here: 

https://github.com/AndreasTraut/Machine-Learning-with-Python/blob/master/Movies%20Machine%20Learning%20-%20Predict%20NaNs.ipynb


## 2. Step-by-step Python-Code for Machine Learning

As Jupyter Notebooks are not always the best environment for Python coding (e.g. Debugging), I extracted the most essential parts of Chapter 2 of Aurelien Geron "Machine Learning with Scikit-Learn & Tensorflow", sorted and arranged the code fragments slightly and created the following structured Python code for being used for example in Spyder (https://www.spyder-ide.org/). The structure of the Python code is a bit similar to the steps, which I followed in the Jupyter Notebooks above and are as follows (you will find these sections also in the ".py" file): 

 1. create index:    
	 1.1 Alternative 1: generate id with static data
	 1.2 Alternative 2: generate stratified sampling
	 1.3 verify if stratified example is good

 2. Discover and visualize the data to gain insights

 3. prepare for Machine Learning
	 3.1 find all NULL-values
	 3.2 remove all NULL-values

 4. Use "Imputer" to clean NaNs

 5. treat "categorial" inputs

 6. custom transformer and pipelines
	 6.1 custom transformer
	 6.2 pipelines

 7. select and train model
	 7.1 LinearRegression model
	 7.2 DecisionTreeRegressor model

 8. crossvalidation 
	 8.1 for DecisionTreeRegressor
	 8.2 for LinearRegression
	 8.3 for RandomForestRegressor
	 8.4 for ExtraTreesRegressor

 9. Save Model

 10. Optimize Model
	 10.1 GridSearchCV
		 10.1.1 GridSearchCV on RandomForestRegressor
		 10.1.2 GridSearchCV on LinearRegressor
	 10.2 Randomized Search
	 10.3 Analyze best models

 11. Evaluate final model on test dataset
