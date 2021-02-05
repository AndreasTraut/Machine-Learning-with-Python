# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:02:33 2020

This example provides essential code fragments, for Machine Learning in Sklearn. 

It is based on "Chapter 2" of the book "Praxiseinstieg Machine Learning mit 
Scikit-Learn und Tensorflow" from Aurelien Geron. He provided some Jupyter-Notebooks
under the Apache License 2.0, which I sorted, adapted and modified. 

This code does NOT work unless you have Python installed. 

@author: Andreas Traut
@date: 27.03.2020

#%% #######################################################################
# 1. create index   
	# 1.1 Alternative 1: generate id with static data
	# 1.2 Alternative 2: generate stratified sampling
	# 1.3 verify if stratified example is good

# 2. Discover and visualize the data to gain insights

# 3. prepare for Machine Learning
	# 3.1 find all NULL-values
	# 3.2 remove all NULL-values

# 4. Use "Imputer" to clean NaNs

# 5. treat "categorial" inputs

# 6. custom transformer and pipelines
	# 6.1 custom transformer
	# 6.2 pipelines

# 7. select and train model
	# 7.1 LinearRegression model
	# 7.2 DecisionTreeRegressor model

# 8. crossvalidation 
	# 8.1 for DecisionTreeRegressor
	# 8.2 for LinearRegression
	# 8.3 for RandomForestRegressor
	# 8.4 for ExtraTreesRegressor

# 9. Save Model

# 10. Optimize Model
	# 10.1 GridSearchCV
		# 10.1.1 GridSearchCV on RandomForestRegressor
		# 10.1.2 GridSearchCV on LinearRegressor
	# 10.2 Randomized Search
	# 10.3 Analyze best models

# 11. Evaluate final model on test dataset
#%% #######################################################################
"""

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import tarfile
from six.moves import urllib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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


#%%
# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# to make this notebook's output stable across runs
np.random.seed(42)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#%% Download the zip-file # In[4-7]:
# import os
# import tarfile
# from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# fetch_housing_data()
#%% read the csv-file
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head()

#%% #######################################################################
#%% 
# =============================================================================
# # 1. create index # In[8]:    
# =============================================================================
print("\n\n 1. create index # In[8]: \n")

# import hashlib
print(housing.info())
housing_with_id = housing.reset_index()   # adds an `index` column

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

#%%
# 1.1 Alternative 1: generate id with static data
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

#%%
# 1.2 Alternative 2: generate stratified sampling
print("\n1.2 Alternative 2: generate stratified sampling\n")

housing["median_income"].hist()
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts()
housing["income_cat"].hist()

# from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_test_set["income_cat"].value_counts() / len(strat_test_set)
housing["income_cat"].value_counts() / len(housing)

#%% 
# 1.3 verify if stratified example is good  In[29]
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()

compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

print(compare_props)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

#%% #######################################################################
#%% 
# =============================================================================
# # 2. Discover and visualize the data to gain insights # In[32]
# =============================================================================
print("\n\n 2. Discover and visualize the data to gain insights # In[32] \n")

# housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", title="bad_visualization_plot")
save_fig("bad_visualization_plot")

# from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.suptitle("scatter_matrix_plot")
save_fig("scatter_matrix_plot")

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False, title="housing_prices_scatterplot")
plt.legend()
save_fig("housing_prices_scatterplot")

housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, title="income_vs_house_value_scatterplot")
plt.axis([0, 16, 0, 550000])
save_fig("income_vs_house_value_scatterplot")

# create new variables
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2, title="median of rooms_per_household")
plt.axis([0, 5, 0, 520000])
plt.show()


#%% #######################################################################
#%% 
# =============================================================================
# # 3. prepare for Machine Learning In[44]
# =============================================================================
print("\n\n 3. prepare for Machine Learning\n")

housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()

# 3.1 find all NULL-values
print("\n3.1 find all NULL-values\n")

print(housing.info())
print("Are there nans in column total_bedrooms?\n", housing["total_bedrooms"].isnull().any())
print("Show rows with nan:\n", housing[housing["total_bedrooms"].isnull()])

#%%
# 3.2 remove all NULL-values
print("\n3.2 remove all NULL-values \n")

sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
# sample_incomplete_rows.dropna(subset=["total_bedrooms"])    # option 1 # In[54]:
# sample_incomplete_rows.drop("total_bedrooms", axis=1)       # option 2 # In[55]:

median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3 # In[56]:
print("sample_incomplete_rows\n", sample_incomplete_rows)

#%% #######################################################################
#%% 
# =============================================================================
# # 4. Use "Imputer" to clean NaNs  #In[57]:
# =============================================================================
print("\n\n 4. Use Imputer In[57]: \n")
    
try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer

imputer = SimpleImputer(strategy="median")
# Remove all text attributes because median can only be calculated on numerical attributes:
housing_num = housing.select_dtypes(include=[np.number]) #or: housing_num = housing.drop('ocean_proximity', axis=1) 
imputer.fit(housing_num)
print("imputer.strategy\n", imputer.strategy)
print("imputer.statistics_\n", imputer.statistics_)
print("housing_num.median\n", housing_num.median().values)  # Check that this is the same as manually computing the median of each attribute:
print("housing_num.mean\n", housing_num.mean().values)  # Check that this is the same as manually computing the median of each attribute:
X = imputer.transform(housing_num) # Transform the training set:
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)
housing_tr.loc[sample_incomplete_rows.index.values]

#%% #######################################################################
#%% 
# =============================================================================
# # 5. treat "categorial" inputs  #In[67]
# =============================================================================
print("\n\n 5. treat categorial inputs  #In[67] \n")

housing_cat = housing[['ocean_proximity']]
print("housing_cat.head\n", housing_cat.head(10), "\n")
try:
    from sklearn.preprocessing import OrdinalEncoder # just to raise an ImportError if Scikit-Learn < 0.20
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    from future_encoders import OneHotEncoder # Scikit-Learn < 0.20

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print("cat_encoder.categories_:\n", cat_encoder.categories_)
print("housing_cat_1hot.toarray():\n", housing_cat_1hot.toarray())
print("housing_cat_1hot:\n", housing_cat_1hot)

#%% #######################################################################
#%% 
# =============================================================================
# # 6. custom transformer and pipelines # In[67]
# =============================================================================
print("\n\n 6. custom transformer and pipelines # In[67] \n")

# 6.1 custom transformer # In[67]
print("6.1 custom transformer #In[67] \n")

print("housing.columns\n", housing.columns)
rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]

def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

# from sklearn.preprocessing import FunctionTransformer
attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                 kw_args={"add_bedrooms_per_room": False})
housing_extra_attribs = attr_adder.fit_transform(housing.values)

housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
print("housing_extra_attribs.head()\n", housing_extra_attribs.head())

#%%
# 6.2 pipelines
print("6.2 pipelines \n")

# Now let's build a pipeline for preprocessing the numerical attributes 
# (note that we could use `CombinedAttributesAdder()` instead
# of `FunctionTransformer(...)` if we preferred):
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, 
                                              validate=False)),
        ('std_scaler', StandardScaler()),
    ])
housing_num_tr = num_pipeline.fit_transform(housing_num)
print("housing_num_tr\n", housing_num_tr)

try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    from future_encoders import ColumnTransformer # Scikit-Learn < 0.20

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
housing_prepared = full_pipeline.fit_transform(housing)
print("housing_prepared\n", housing_prepared)

#%% #######################################################################
#%%
# =============================================================================
# # 7. select and train model #In[82]
# =============================================================================
print("\n\n 7. select and train model #In[82]\n")

# 7.1 LinearRegression model
print("7.1 LinearRegression model\n")

# from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
# let's try the full preprocessing pipeline on a few training instances
some_data = housing.iloc[:1]
some_labels = housing_labels.iloc[:1]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:\n", lin_reg.predict(some_data_prepared))
print("Labels:\n", list(some_labels)) # Compare against the actual values:

# from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("lin_rmse\n", lin_rmse)
#%% 
# 7.2 DecisionTreeRegressor model
print("7.2 DecisionTreeRegressor model\n")
# from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)

tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("tree_rmse\n", tree_rmse)

#%% #######################################################################
#%%
# =============================================================================
# # 8. crossvalidation 
# =============================================================================
print("\n\n 8. crossvalidation \n")

# 8.1 for DecisionTreeRegressor
print("8.1 for DecisionTreeRegressor\n")

# from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

#%%
# 8.2 for LinearRegression
print("8.2 for LinearRegression\n")
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

#%%
# 8.3 for RandomForestRegressor
print("8.3 for RandomForestRegressor\n")

# from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)
# from sklearn.model_selection import cross_val_score
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

#%%
# 8.4 for ExtraTreesRegressor
print("8.4 for ExtraTreesRegressor\n")

from sklearn.ensemble import ExtraTreesRegressor
extratree_reg = ExtraTreesRegressor(n_estimators=10, random_state=42)
extratree_reg.fit(housing_prepared, housing_labels)

housing_predictions = extratree_reg.predict(housing_prepared)
extratree_mse = mean_squared_error(housing_labels, housing_predictions)
extratree_rmse = np.sqrt(extratree_mse)
print(extratree_rmse)
extratree_scores = cross_val_score(extratree_reg, housing_prepared, 
                                   housing_labels, 
                                   scoring = "neg_mean_squared_error", cv=10)
extratree_rmse_scores = np.sqrt(-extratree_scores)
display_scores(extratree_rmse_scores)

#%% #######################################################################
#%% 
# =============================================================================
# # 9. Save Model
# =============================================================================
print("\n\n 9. Save Model\n")

# from sklearn.externals import joblib
joblib.dump(forest_reg, "forest_reg.pkl")
# und später...
my_model_loaded = joblib.load("forest_reg.pkl")

#%% #######################################################################
#%% 
# =============================================================================
# # 10. Optimize Model # In[98]
# =============================================================================
print("\n\n 10. Optimize Model # In[98]\n")

# 10.1 GridSearchCV
print("\n 10.1 GridSearchCV on RandomForestRegressor\n")


#%%
# 10.1.1 GridSearchCV on RandomForestRegressor
print("\n 10.1.1 GridSearchCV on RandomForestRegressor\n")

# from sklearn.model_selection import GridSearchCV
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [30, 40, 50], 'max_features': [2, 4, 6, 8, 10]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', 
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
a= pd.DataFrame(grid_search.cv_results_)
#%%
# 10.1.2 GridSearchCV on LinearRegressor
print("\n 10.1.1 GridSearchCV on LinearRegressor\n")

# from sklearn.model_selection import GridSearchCV
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'fit_intercept': [True], 'n_jobs': [2, 4, 6, 8, 10]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'normalize': [False], 'n_jobs': [3, 10]},
  ]

lin_reg = LinearRegression()
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
lin_grid_search = GridSearchCV(lin_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', 
                           return_train_score=True)
lin_grid_search.fit(housing_prepared, housing_labels)
# print(lin_grid_search.best_params_)
print(lin_grid_search.best_estimator_)
cvres = lin_grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
a= pd.DataFrame(lin_grid_search.cv_results_)

#%%
# 10.2 Randomized Search
print("\n 10.2 Randomized Search\n")

# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint
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
rnd_search.fit(housing_prepared, housing_labels)
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#%%
# 10.3 Analyze best models # In[105]
print("\n 10.3 Analyze best models # In[105]\n")

feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
#cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

#%% #######################################################################
#%%
# =============================================================================
# # 11. Evaluate final model on test dataset
# =============================================================================
print("\n\n 11. Evaluate final model on test dataset\n")

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print ("final_predictions\n", final_predictions )
print ("final_rmse \n", final_rmse )

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
mean = squared_errors.mean()
m = len(squared_errors)

# from scipy import stats
print("95% confidence interval: ", 
      np.sqrt(stats.t.interval(confidence, m - 1,
                         loc=np.mean(squared_errors),
                         scale=stats.sem(squared_errors)))
      )
