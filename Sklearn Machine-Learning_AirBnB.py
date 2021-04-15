# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:02:33 2020

@author: Andreas Traut
@date: 07.04.2021

#%% #######################################################################
# 1. Initialize and Read the CSV File

# S. Split Training Data and Test Data  
	# S.1 Alternative 1: generate id with static data
	# S.2 Alternative 2: generate stratified sampling
	# S.3 verify if stratified sample is good

# 2. Discover and Visualize the Data to Gain Insights

# 3. Clean NULL-Values and Prepare for Machine Learning
	# 3.1 find all NULL-values
	# 3.2 remove all NULL-values

# 4. Model-Specific Preprocessing
    # 4.1 Use "Imputer" to clean NaNs
    # 4.2 Treat "Categorial" Inputs

# 5. Pipelines and Custom Transformer
	# 5.1 Custom Transformer
	# 5.2 Pipelines

# 6. Select and Train Model
	# 6.1 LinearRegression model
	# 6.2 DecisionTreeRegressor model

# 7. Crossvalidation 
	# 7.1 for DecisionTreeRegressor
	# 7.2 for LinearRegression
	# 7.3 for RandomForestRegressor
	# 7.4 for ExtraTreesRegressor

# 8. Save Model

# 9. Optimize Model
	# 9.1 GridSearchCV
		# 9.1.1 GridSearchCV on RandomForestRegressor
		# 9.1.2 GridSearchCV on LinearRegressor
	# 9.2 Randomized Search
	# 9.3 Analyze Best Models

# 10. Evaluate Final Model on Test Dataset
#%% #######################################################################
"""

# Common imports
from __future__ import division, print_function, unicode_literals
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import hashlib
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from scipy import stats

#%% #######################################################################
#
# =============================================================================
# # 1. initialize and read the file
# =============================================================================
#%%
# Where to save the figures
PROJECT_ROOT_DIR = "."
myDataset_NAME = "AirBnB"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "media")
myDataset_PATH = os.path.join("datasets", "AirBnB")

def save_fig(fig_id, prefix=myDataset_NAME, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, prefix + "_" + fig_id + "." + fig_extension)
    print("Saving figure", prefix + "_" + fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
np.random.seed(42)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#%% read the csv-file
def load_myDataset_data(myDataset_path=myDataset_PATH):
    csv_path = os.path.join(myDataset_path, "listings.csv")
    return pd.read_csv(csv_path)

myDataset = load_myDataset_data()
print(myDataset.head())

#%% remove unwanted columns

myDataset = myDataset.drop("id", axis=1)  

#%% #######################################################################
#
# =============================================================================
# # 1. Split Training Data and Test Data 
# =============================================================================
print("\n\n1. create index\n")

print(myDataset.info())
myDataset_with_id = myDataset.reset_index()
print(myDataset_with_id.head())

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

#%%
# S.1 Alternative 1: generate id with static data
print("\n1.1 Alternative 1: generate id with static data\n")

myDataset_with_id["index"] = round(myDataset["longitude"],2) * 10000 + myDataset["latitude"]
train_set, test_set = split_train_test_by_id(myDataset_with_id, 0.2, "index")
print(myDataset_with_id.head())
print("train set: {0:7d}\ntest set : {1:7d}".format(len(train_set),len(test_set)))

#%%
# S.2 Alternative 2: generate stratified sampling
# Requirement:  from sklearn.model_selection import StratifiedShuffleSplit
print("\n1.2 Alternative 2: generate stratified sampling\n")

myDataset["price"].hist()
myDataset["price_cat"] = pd.cut(myDataset["price"],
                               bins=[-1, 50, 100, 200, 400, np.inf],
                               labels=[50, 100, 200, 400, 500])
print("\nvalue_counts\n", myDataset["price_cat"].value_counts())
myDataset["price_cat"].hist()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(myDataset, myDataset["price_cat"]):
    strat_train_set = myDataset.loc[train_index]
    strat_test_set = myDataset.loc[test_index]

print("\nstrat_test_set\n", strat_test_set["price_cat"].value_counts() / len(strat_test_set))
print("\nmyDataset\n", myDataset["price_cat"].value_counts() / len(myDataset))

#%% 
# S.3 verify if stratified example is good 
print("\n1.3 verify if stratified sample is good \n")

def price_cat_proportions(data):
    return data["price_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(myDataset, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": price_cat_proportions(myDataset),
    "Stratified": price_cat_proportions(strat_test_set),
    "Random": price_cat_proportions(test_set),
}).sort_index()

compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

print(compare_props)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("price_cat", axis=1, inplace=True)

#%% #######################################################################
#
# =============================================================================
# # 2. Discover and Visualize the Data to Gain Insights
# =============================================================================
# Requirement: from pandas.plotting import scatter_matrix
print("\n\n2. Discover and visualize the data to gain insights \n")

# myDataset = strat_train_set.copy()
myDataset.plot(kind="scatter", x="longitude", y="latitude", 
               title="bad_visualization_plot")
save_fig("bad_visualization_plot")

attributes = ["price", "number_of_reviews", "host_id", "availability_365", 
              "reviews_per_month", "minimum_nights"]
scatter_matrix(myDataset[attributes], figsize=(12, 8))
plt.suptitle("scatter_matrix_plot")
save_fig("scatter_matrix_plot")
#%%
# myDataset = myDataset[(myDataset['longitude']>=13.32) & (myDataset['longitude']<=13.35)]
myDataset.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=myDataset['price']/100, label="price", figsize=(10,7),
    c="price", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False, title="prices_scatterplot")
plt.legend()
save_fig("prices_scatterplot")
#%%
corr_matrix = myDataset.corr()
print("correlation:\n", corr_matrix["price"].sort_values(ascending=False))

#%% #######################################################################
#
# =============================================================================
# # 3. Clean NULL-Values and Prepare for Machine Learning
# =============================================================================
print("\n\n3. prepare for Machine Learning\n")

myDataset = strat_train_set.drop("price", axis=1) # drop labels for training set
myDataset_labels = strat_train_set["price"].copy()

# 3.1 find all NULL-values
print("\n3.1 find all NULL-values\n")

print("\nHow many Non-NULLrows are there?\n")
print(myDataset.info())
print("\nAre there NULL values in the columns?\n", myDataset.isnull().any())
print("\nAre there NULLs in column reviews_per_month?\n", myDataset["reviews_per_month"].isnull().any())
print("\nShow some rows with NULL (head only):\n", myDataset[myDataset["reviews_per_month"].isnull()].head())

#%%
# 3.2 remove all NULL-values
print("\n3.2 remove all NULL-values \n")

sample_incomplete_rows = myDataset[myDataset.isnull().any(axis=1)] 

# option 1: remove rows which contains NaNs
# sample_incomplete_rows.dropna(subset=["total_bedrooms"])    

# option 2 : remove columns with contain NaNs
# sample_incomplete_rows.drop("total_bedrooms", axis=1)       

# option 3 : replace NaN by median
median = myDataset["reviews_per_month"].median()
sample_incomplete_rows["reviews_per_month"].fillna(median, inplace=True) 

print("sample_incomplete_rows\n", sample_incomplete_rows['reviews_per_month'].head())

#%% #######################################################################
#
# =============================================================================
# # 4. Model-Specific Preprocessing
# =============================================================================

# 4.1 Use "Imputer" to Clean NaN
# Requirement:  from sklearn.impute import SimpleImputer 
print("\n\n4.1. Use Imputer to Clean NaN\n")

imputer = SimpleImputer(strategy="median")
myDataset_num = myDataset.select_dtypes(include=[np.number]) #or: myDataset_num = myDataset.drop('ocean_proximity', axis=1) 
imputer.fit(myDataset_num)
print("\nimputer.strategy\n", imputer.strategy)
print("\nimputer.statistics_\n", imputer.statistics_)
print("\nmyDataset_num.median\n", myDataset_num.median().values)  # Check that this is the same as manually computing the median of each attribute:
print("\nmyDataset_num.mean\n", myDataset_num.mean().values)  # Check that this is the same as manually computing the median of each attribute:
X = imputer.transform(myDataset_num) # Transform the training set:
myDataset_tr = pd.DataFrame(X, columns=myDataset_num.columns,
                          index=myDataset.index)
myDataset_tr.loc[sample_incomplete_rows.index.values]

#%%
# 4.2 Treat "Categorial" Inputs
# Requirement: from sklearn.preprocessing import OneHotEncoder
print("\n\n4.1. Treat Categorial Inputs\n")


myDataset_cat = myDataset[['room_type']]
print("myDataset_cat.head\n", myDataset_cat.head(10), "\n")

cat_encoder = OneHotEncoder()
myDataset_cat_1hot = cat_encoder.fit_transform(myDataset_cat)
print("\ncat_encoder.categories_:\n", cat_encoder.categories_)
print("\nmyDataset_cat_1hot.toarray():\n", myDataset_cat_1hot.toarray())
print("\nmyDataset_cat_1hot:\n", myDataset_cat_1hot)

#%% #######################################################################
#
# =============================================================================
# # 5. Pipelines and Custom Transformer
# =============================================================================
print("\n\n 5. custom transformer and pipelines \n")

# 5.1 custom transformer
# Requirement: from sklearn.preprocessing import FunctionTransformer
print("5.1 custom transformer\n")

print("myDataset.columns\n", myDataset_num.columns)
number_of_reviews_ix, availability_365_ix, calculated_host_listings_count_ix, reviews_per_month_ix = [
    list(myDataset_num.columns).index(col)
    for col in ("number_of_reviews", "availability_365", "calculated_host_listings_count", "reviews_per_month")]

def add_extra_features(X):
    number_reviews_dot_revievs_per_month = X[:, number_of_reviews_ix] * X[:, reviews_per_month_ix]
    return np.c_[X, number_reviews_dot_revievs_per_month]

attr_adder = FunctionTransformer(add_extra_features, validate=False)
myDataset_extra_attribs = attr_adder.fit_transform(myDataset_num.values)

myDataset_extra_attribs = pd.DataFrame(
    myDataset_extra_attribs,
    columns=list(myDataset_num.columns)+["number_reviews_dot_revievs_per_month"],
    index=myDataset_num.index)
print("myDataset_extra_attribs.head()\n", myDataset_extra_attribs.head())

#%%
# 5.2 Pipelines
# Requirements: from sklearn.pipeline import Pipeline
#               from sklearn.preprocessing import StandardScaler
#               from sklearn.compose import ColumnTransformer
print("5.2 Pipelines \n")

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, 
                                              validate=False)),
        ('std_scaler', StandardScaler())
        ])
myDataset_num_tr = num_pipeline.fit_transform(myDataset_num)
print("myDataset_num_tr\n", myDataset_num_tr)

num_attribs = list(myDataset_num)
cat_attribs = ["room_type"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
myDataset_prepared = full_pipeline.fit_transform(myDataset)
print("myDataset_prepared\n", myDataset_prepared)

#%% #######################################################################
#
# =============================================================================
# # 6. Select and Train Model
# =============================================================================
print("\n\n6. select and train model\n")

# 6.1 LinearRegression model
# Requirement:  from sklearn.linear_model import LinearRegression
#               from sklearn.metrics import mean_squared_error
print("6.1 LinearRegression model\n")

lin_reg = LinearRegression()
lin_reg.fit(myDataset_prepared, myDataset_labels)
some_data = myDataset.iloc[:10]
some_labels = myDataset_labels.iloc[:10]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:\n", lin_reg.predict(some_data_prepared))
print("Labels:\n", list(some_labels)) # Compare against the actual values:

myDataset_predictions = lin_reg.predict(myDataset_prepared)
lin_mse = mean_squared_error(myDataset_labels, myDataset_predictions)
lin_rmse = np.sqrt(lin_mse)
print("lin_rmse\n", lin_rmse)

print("mean of labels:\n", myDataset_labels.mean())
print("std deviation of labels:\n", myDataset_labels.std())
myDataset_labels.hist()

#%% 
# 6.2 DecisionTreeRegressor Model
# Requirement:  from sklearn.tree import DecisionTreeRegressor
print("6.2 DecisionTreeRegressor Model\n")

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(myDataset_prepared, myDataset_labels)
myDataset_predictions = tree_reg.predict(myDataset_prepared)

tree_mse = mean_squared_error(myDataset_labels, myDataset_predictions)
tree_rmse = np.sqrt(tree_mse)
print("tree_rmse\n", tree_rmse)

#%% #######################################################################
#
# =============================================================================
# # 7. Crossvalidation 
# =============================================================================
print("\n\n8. crossvalidation \n")

# 7.1 for DecisionTreeRegressor
# Requirement:  from sklearn.model_selection import cross_val_score
print("7.1 for DecisionTreeRegressor\n")

scores = cross_val_score(tree_reg, myDataset_prepared, 
                         myDataset_labels,
                         scoring="neg_mean_squared_error", 
                         cv=10)
tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

#%%
# 7.2 for LinearRegression
print("7.2 for LinearRegression\n")

lin_scores = cross_val_score(lin_reg, myDataset_prepared, 
                             myDataset_labels,
                             scoring="neg_mean_squared_error", 
                             cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

#%%
# 7.3 for RandomForestRegressor
# Requirements: from sklearn.ensemble import RandomForestRegressor
#               from sklearn.model_selection import cross_val_score
print("7.3 for RandomForestRegressor\n")

forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(myDataset_prepared, myDataset_labels)

myDataset_predictions = forest_reg.predict(myDataset_prepared)
forest_mse = mean_squared_error(myDataset_labels, myDataset_predictions)
forest_rmse = np.sqrt(forest_mse)
print("forest_rmse\n", forest_rmse)

forest_scores = cross_val_score(forest_reg, myDataset_prepared, 
                                myDataset_labels,
                                scoring="neg_mean_squared_error", 
                                cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

#%%
# 7.4 for ExtraTreesRegressor
print("7.4 for ExtraTreesRegressor\n")

from sklearn.ensemble import ExtraTreesRegressor
extratree_reg = ExtraTreesRegressor(n_estimators=10, 
                                    random_state=42)
extratree_reg.fit(myDataset_prepared, myDataset_labels)

myDataset_predictions = extratree_reg.predict(myDataset_prepared)
extratree_mse = mean_squared_error(myDataset_labels, myDataset_predictions)
extratree_rmse = np.sqrt(extratree_mse)
print("extratree_rmse\n", extratree_rmse)

extratree_scores = cross_val_score(extratree_reg, 
                                   myDataset_prepared, 
                                   myDataset_labels, 
                                   scoring = "neg_mean_squared_error", 
                                   cv=10)
extratree_rmse_scores = np.sqrt(-extratree_scores)
display_scores(extratree_rmse_scores)

#%% #######################################################################
#
# =============================================================================
# # 8. Save Model
# =============================================================================
# Requirement: import joblib
print("\n\n8. Save Model\n")

joblib.dump(forest_reg, "forest_reg.pkl")
# und später...
my_model_loaded = joblib.load("forest_reg.pkl")

#%% #######################################################################
#
# =============================================================================
# # 9. Optimize Model
# =============================================================================
print("\n\n9. Optimize Model\n")

# 9.1 GridSearchCV
print("\n9.1 GridSearchCV on RandomForestRegressor\n")

# 9.1.1 GridSearchCV on RandomForestRegressor
# Requirement:  from sklearn.model_selection import GridSearchCV
print("\n9.1.1 GridSearchCV on RandomForestRegressor\n")

param_grid = [
    {'n_estimators': [30, 40, 50], 'max_features': [2, 4, 6, 8, 10]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
  ]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', 
                           return_train_score=True)
grid_search.fit(myDataset_prepared, myDataset_labels)

print("Best Params: ", grid_search.best_params_)
print("Best Estimator: ", grid_search.best_estimator_)
print("\nResults (mean_test_score and params):")
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
print("\nResults (complete):")
a = pd.DataFrame(grid_search.cv_results_)
print(a)

#%%
# 9.1.2 GridSearchCV on LinearRegressor
# Requirement:  from sklearn.model_selection import GridSearchCV
print("\n9.1.1 GridSearchCV on LinearRegressor\n")

param_grid = [
    {'fit_intercept': [True], 'n_jobs': [2, 4, 6, 8, 10]},
    {'normalize': [False], 'n_jobs': [3, 10]},
  ]

lin_reg = LinearRegression()
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
lin_grid_search = GridSearchCV(lin_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', 
                           return_train_score=True)
lin_grid_search.fit(myDataset_prepared, myDataset_labels)

print("Best Params: ", lin_grid_search.best_params_)
print("Best Estimator: ", lin_grid_search.best_estimator_)
print("\nResults (mean_test_score and params):")
cvres = lin_grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
print("\nResults (complete):")
a = pd.DataFrame(lin_grid_search.cv_results_)
print(a)

#%%
# 9.2 Randomized Search
# Requirements: from sklearn.model_selection import RandomizedSearchCV
#               from scipy.stats import randint
print("\n9.2 Randomized Search\n")

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, 
                                param_distributions=param_distribs,
                                n_iter=10, cv=5, 
                                scoring='neg_mean_squared_error', 
                                random_state=42)
rnd_search.fit(myDataset_prepared, myDataset_labels)

print("Best Params: ", rnd_search.best_params_)
print("Best Estimator: ", rnd_search.best_estimator_)
print("\nResults (mean_test_score and params):")
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

print("\nResults (complete):")
a = pd.DataFrame(rnd_search.cv_results_)
print(a)

#%%
# 9.3 Analyze best models
print("\n9.3 Analyze best models\n")

feature_importances = grid_search.best_estimator_.feature_importances_
print("feature_importances:\n", feature_importances)
extra_attribs = ["number_reviews_dot_revievs_per_month"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print("\nattributes:\n", attributes)
my_list = sorted(zip(feature_importances, attributes), reverse=True)
print("\nMost important features (think about removing features):")
print("\n".join('{}' for _ in range(len(my_list))).format(*my_list))

#%% #######################################################################
#
# =============================================================================
# # 10. Evaluate final model on test dataset
# =============================================================================
# Requirment: from scipy import stats
print("\n\n 10. Evaluate final model on test dataset\n")

final_model = grid_search.best_estimator_
print("final_model:\n", final_model)

X_test = strat_test_set.drop("price", axis=1)
y_test = strat_test_set["price"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print ("final_predictions:\n", final_predictions )
print ("final_rmse:\n", final_rmse )

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
mean = squared_errors.mean()
m = len(squared_errors)


print("95% confidence interval: ", 
      np.sqrt(stats.t.interval(confidence, m - 1,
                         loc=np.mean(squared_errors),
                         scale=stats.sem(squared_errors)))
      )



side_by_side = [(true, pred, (true-pred)/true)
                for true, pred in
                zip(list(y_test),
                    list(final_predictions))]
print(side_by_side)
test_set.insert(loc=1, column="final_prediction", value = final_predictions)
print(test_set)
