<map version="1.0.1">
<!-- To view this file, download free mind mapping software FreeMind from http://freemind.sourceforge.net -->
<node CREATED="1595227154339" LINK="Data%20Scientist%20-%20Python%20%20Jupyter.mm" MODIFIED="1595252134435" TEXT="Programmablauf (SkLearn)">
<node CREATED="1595227169028" MODIFIED="1595251422436" POSITION="right" TEXT="1. import and create index">
<node CREATED="1595231267926" MODIFIED="1595231319527" TEXT="os.path">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    </p>
    <p>
      &#160;&#160;&#160;&#160;os.makedirs(housing_path, exist_ok=True)
    </p>
    <p>
      &#160;&#160;&#160;&#160;tgz_path = os.path.join(housing_path, &quot;housing.tgz&quot;)
    </p>
    <p>
      &#160;&#160;&#160;&#160;urllib.request.urlretrieve(housing_url, tgz_path)
    </p>
    <p>
      &#160;&#160;&#160;&#160;housing_tgz = tarfile.open(tgz_path)
    </p>
    <p>
      &#160;&#160;&#160;&#160;housing_tgz.extractall(path=housing_path)
    </p>
    <p>
      &#160;&#160;&#160;&#160;housing_tgz.close()
    </p>
    <p>
      
    </p>
    <p>
      def load_housing_data(housing_path=HOUSING_PATH):
    </p>
    <p>
      &#160;&#160;&#160;&#160;csv_path = os.path.join(housing_path, &quot;housing.csv&quot;)
    </p>
    <p>
      &#160;&#160;&#160;&#160;return pd.read_csv(csv_path)
    </p>
  </body>
</html></richcontent>
</node>
<node CREATED="1595231329965" MODIFIED="1595231390613" TEXT="xx_with_id">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      housing_with_id[&quot;id&quot;] = housing[&quot;longitude&quot;] * 1000 + housing[&quot;latitude&quot;]
    </p>
    <p>
      
    </p>
    <p>
      
    </p>
  </body>
</html></richcontent>
</node>
<node CREATED="1595231455061" MODIFIED="1595231486625" TEXT="stratified samples">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      # from sklearn.model_selection import StratifiedShuffleSplit
    </p>
    <p>
      split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    </p>
    <p>
      for train_index, test_index in split.split(housing, housing[&quot;income_cat&quot;]):
    </p>
    <p>
      &#160;&#160;&#160;&#160;strat_train_set = housing.loc[train_index]
    </p>
    <p>
      &#160;&#160;&#160;&#160;strat_test_set = housing.loc[test_index]
    </p>
    <p>
      
    </p>
    <p>
      strat_test_set[&quot;income_cat&quot;].value_counts() / len(strat_test_set)
    </p>
  </body>
</html></richcontent>
</node>
</node>
<node CREATED="1595231123423" MODIFIED="1595251422438" POSITION="right" TEXT="2. discover and visualize data">
<node CREATED="1595231497222" MODIFIED="1595231532262" TEXT="plot">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      # from pandas.plotting import scatter_matrix
    </p>
    <p>
      attributes = [&quot;median_house_value&quot;, &quot;median_income&quot;, &quot;total_rooms&quot;,
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&quot;housing_median_age&quot;]
    </p>
    <p>
      scatter_matrix(housing[attributes], figsize=(12, 8))
    </p>
    <p>
      plt.suptitle(&quot;scatter_matrix_plot&quot;)
    </p>
    <p>
      save_fig(&quot;scatter_matrix_plot&quot;)
    </p>
    <p>
      
    </p>
    <p>
      housing.plot(kind=&quot;scatter&quot;, x=&quot;longitude&quot;, y=&quot;latitude&quot;, alpha=0.4,
    </p>
    <p>
      &#160;&#160;&#160;&#160;s=housing[&quot;population&quot;]/100, label=&quot;population&quot;, figsize=(10,7),
    </p>
    <p>
      &#160;&#160;&#160;&#160;c=&quot;median_house_value&quot;, cmap=plt.get_cmap(&quot;jet&quot;), colorbar=True,
    </p>
    <p>
      &#160;&#160;&#160;&#160;sharex=False, title=&quot;housing_prices_scatterplot&quot;)
    </p>
  </body>
</html></richcontent>
</node>
<node CREATED="1595231566238" MODIFIED="1595231590745" TEXT="new variables">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      housing[&quot;rooms_per_household&quot;] = housing[&quot;total_rooms&quot;]/housing[&quot;households&quot;]
    </p>
    <p>
      housing[&quot;bedrooms_per_room&quot;] = housing[&quot;total_bedrooms&quot;]/housing[&quot;total_rooms&quot;]
    </p>
    <p>
      housing[&quot;population_per_household&quot;]=housing[&quot;population&quot;]/housing[&quot;households&quot;]
    </p>
  </body>
</html></richcontent>
</node>
<node CREATED="1595231532254" MODIFIED="1595231563726" TEXT="correlation matrix">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      corr_matrix = housing.corr()
    </p>
    <p>
      corr_matrix[&quot;median_house_value&quot;].sort_values(ascending=False)
    </p>
  </body>
</html></richcontent>
</node>
</node>
<node CREATED="1595231134428" MODIFIED="1595251422441" POSITION="right" TEXT="3. - 5. prepare and clean data">
<node CREATED="1595231599047" MODIFIED="1595231740538" TEXT="remove NULL values">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
    </p>
    <p>
      # sample_incomplete_rows.dropna(subset=[&quot;total_bedrooms&quot;])&#160;&#160;&#160;&#160;# option 1 # In[54]:
    </p>
    <p>
      # sample_incomplete_rows.drop(&quot;total_bedrooms&quot;, axis=1)&#160;&#160;&#160;&#160;&#160;&#160;&#160;# option 2 # In[55]:
    </p>
    <p>
      
    </p>
    <p>
      median = housing[&quot;total_bedrooms&quot;].median()
    </p>
    <p>
      sample_incomplete_rows[&quot;total_bedrooms&quot;].fillna(median, inplace=True) # option 3 # In[56]:
    </p>
  </body>
</html></richcontent>
</node>
<node CREATED="1595231634799" MODIFIED="1595231748658" TEXT="Imputer">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      from sklearn.preprocessing import Imputer as SimpleImputer
    </p>
    <p>
      
    </p>
    <p>
      imputer = SimpleImputer(strategy=&quot;median&quot;)
    </p>
    <p>
      # Remove all text attributes because median can only be calculated on numerical attributes:
    </p>
    <p>
      housing_num = housing.select_dtypes(include=[np.number]) #or: housing_num = housing.drop('ocean_proximity', axis=1)
    </p>
    <p>
      imputer.fit(housing_num)
    </p>
    <p>
      print(&quot;imputer.strategy\n&quot;, imputer.strategy)
    </p>
    <p>
      print(&quot;imputer.statistics_\n&quot;, imputer.statistics_)
    </p>
    <p>
      print(&quot;housing_num.median\n&quot;, housing_num.median().values)&#160;&#160;# Check that this is the same as manually computing the median of each attribute:
    </p>
    <p>
      print(&quot;housing_num.mean\n&quot;, housing_num.mean().values)&#160;&#160;# Check that this is the same as manually computing the median of each attribute:
    </p>
    <p>
      X = imputer.transform(housing_num) # Transform the training set:
    </p>
  </body>
</html></richcontent>
</node>
<node CREATED="1595231664318" MODIFIED="1595231686594" TEXT="treat categorial inputs (OneHotEncoder)">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      &#160;&#160;from future_encoders import OneHotEncoder # Scikit-Learn &lt; 0.20
    </p>
    <p>
      
    </p>
    <p>
      cat_encoder = OneHotEncoder()
    </p>
    <p>
      housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    </p>
  </body>
</html></richcontent>
</node>
</node>
<node CREATED="1595231144599" MODIFIED="1595251422442" POSITION="right" TEXT="6. custom transformation and pipelines">
<node CREATED="1595231712303" MODIFIED="1595231726609" TEXT="FunctionTransformer">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      def add_extra_features(X, add_bedrooms_per_room=True):
    </p>
    <p>
      &#160;&#160;&#160;&#160;rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    </p>
    <p>
      &#160;&#160;&#160;&#160;population_per_household = X[:, population_ix] / X[:, household_ix]
    </p>
    <p>
      &#160;&#160;&#160;&#160;if add_bedrooms_per_room:
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;return np.c_[X, rooms_per_household, population_per_household,
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;bedrooms_per_room]
    </p>
    <p>
      &#160;&#160;&#160;&#160;else:
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;return np.c_[X, rooms_per_household, population_per_household]
    </p>
    <p>
      
    </p>
    <p>
      # from sklearn.preprocessing import FunctionTransformer
    </p>
    <p>
      attr_adder = FunctionTransformer(add_extra_features, validate=False,
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;kw_args={&quot;add_bedrooms_per_room&quot;: False})
    </p>
    <p>
      housing_extra_attribs = attr_adder.fit_transform(housing.values)
    </p>
  </body>
</html></richcontent>
</node>
<node CREATED="1595231728158" MODIFIED="1595231787452" TEXT="Pipeline">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      num_pipeline = Pipeline([
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;('imputer', SimpleImputer(strategy=&quot;median&quot;)),
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;('attribs_adder', FunctionTransformer(add_extra_features,
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;validate=False)),
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;('std_scaler', StandardScaler()),
    </p>
    <p>
      &#160;&#160;&#160;&#160;])
    </p>
    <p>
      housing_num_tr = num_pipeline.fit_transform(housing_num)
    </p>
    <p>
      
    </p>
    <p>
      
    </p>
    <p>
      from future_encoders import ColumnTransformer # Scikit-Learn &lt; 0.20
    </p>
    <p>
      
    </p>
    <p>
      num_attribs = list(housing_num)
    </p>
    <p>
      cat_attribs = [&quot;ocean_proximity&quot;]
    </p>
    <p>
      
    </p>
    <p>
      full_pipeline = ColumnTransformer([
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;(&quot;num&quot;, num_pipeline, num_attribs),
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;(&quot;cat&quot;, OneHotEncoder(), cat_attribs),
    </p>
    <p>
      &#160;&#160;&#160;&#160;])
    </p>
    <p>
      housing_prepared = full_pipeline.fit_transform(housing)
    </p>
    <p>
      
    </p>
  </body>
</html></richcontent>
</node>
</node>
<node CREATED="1595231157381" MODIFIED="1595251422443" POSITION="right" TEXT="7. select and train model">
<node CREATED="1595231849327" MODIFIED="1595231869040" TEXT="linear regression">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      lin_reg = LinearRegression()
    </p>
    <p>
      lin_reg.fit(housing_prepared, housing_labels)
    </p>
    <p>
      housing_predictions = lin_reg.predict(housing_prepared)
    </p>
    <p>
      
    </p>
  </body>
</html></richcontent>
</node>
<node CREATED="1595231871782" MODIFIED="1595231881078" TEXT="decision tree">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      tree_reg = DecisionTreeRegressor(random_state=42)
    </p>
    <p>
      tree_reg.fit(housing_prepared, housing_labels)
    </p>
    <p>
      
    </p>
  </body>
</html></richcontent>
</node>
</node>
<node CREATED="1595231160959" MODIFIED="1595231893537" POSITION="right" TEXT="8. cross validation">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;scoring=&quot;neg_mean_squared_error&quot;, cv=10)
    </p>
    <p>
      tree_rmse_scores = np.sqrt(-scores)
    </p>
  </body>
</html></richcontent>
</node>
<node CREATED="1595231166062" MODIFIED="1595231914215" POSITION="right" TEXT="9. save model">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      joblib.dump(forest_reg, &quot;forest_reg.pkl&quot;)
    </p>
    <p>
      # und sp&#228;ter...
    </p>
    <p>
      my_model_loaded = joblib.load(&quot;forest_reg.pkl&quot;)
    </p>
  </body>
</html></richcontent>
</node>
<node CREATED="1595231168446" MODIFIED="1595251422445" POSITION="right" TEXT="10 optimize model">
<node CREATED="1595231916486" MODIFIED="1595231975458" TEXT="GridSearch">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      param_grid = [
    </p>
    <p>
      &#160;&#160;&#160;&#160;# try 12 (3&#215;4) combinations of hyperparameters
    </p>
    <p>
      &#160;&#160;&#160;&#160;{'n_estimators': [30, 40, 50], 'max_features': [2, 4, 6, 8, 10]},
    </p>
    <p>
      &#160;&#160;&#160;&#160;# then try 6 (2&#215;3) combinations with bootstrap set as False
    </p>
    <p>
      &#160;&#160;&#160;&#160;{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    </p>
    <p>
      &#160;&#160;]
    </p>
    <p>
      
    </p>
    <p>
      forest_reg = RandomForestRegressor(random_state=42)
    </p>
    <p>
      # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    </p>
    <p>
      grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;scoring='neg_mean_squared_error',
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;return_train_score=True)
    </p>
    <p>
      grid_search.fit(housing_prepared, housing_labels)
    </p>
  </body>
</html></richcontent>
</node>
<node CREATED="1595231970038" MODIFIED="1595232012891" TEXT="RandomizedSearch">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      param_distribs = {
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;'n_estimators': randint(low=1, high=200),
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;'max_features': randint(low=1, high=8),
    </p>
    <p>
      &#160;&#160;&#160;&#160;}
    </p>
    <p>
      
    </p>
    <p>
      forest_reg = RandomForestRegressor(random_state=42)
    </p>
    <p>
      rnd_search = RandomizedSearchCV(forest_reg,
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;param_distributions=param_distribs,
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;n_iter=10, cv=5,
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;scoring='neg_mean_squared_error',
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;random_state=42)
    </p>
    <p>
      rnd_search.fit(housing_prepared, housing_labels)
    </p>
  </body>
</html></richcontent>
</node>
<node CREATED="1595232012880" MODIFIED="1595232037225" TEXT="Analyze beste model">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      feature_importances = grid_search.best_estimator_.feature_importances_
    </p>
    <p>
      feature_importances
    </p>
    <p>
      extra_attribs = [&quot;rooms_per_hhold&quot;, &quot;pop_per_hhold&quot;, &quot;bedrooms_per_room&quot;]
    </p>
    <p>
      #cat_encoder = cat_pipeline.named_steps[&quot;cat_encoder&quot;] # old solution
    </p>
    <p>
      cat_encoder = full_pipeline.named_transformers_[&quot;cat&quot;]
    </p>
    <p>
      cat_one_hot_attribs = list(cat_encoder.categories_[0])
    </p>
    <p>
      attributes = num_attribs + extra_attribs + cat_one_hot_attribs
    </p>
    <p>
      sorted(zip(feature_importances, attributes), reverse=True)
    </p>
  </body>
</html></richcontent>
</node>
</node>
<node CREATED="1595231176253" MODIFIED="1595232064597" POSITION="right" TEXT="11. evaluate final result">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      final_model = grid_search.best_estimator_
    </p>
    <p>
      
    </p>
    <p>
      X_test = strat_test_set.drop(&quot;median_house_value&quot;, axis=1)
    </p>
    <p>
      y_test = strat_test_set[&quot;median_house_value&quot;].copy()
    </p>
    <p>
      
    </p>
    <p>
      X_test_prepared = full_pipeline.transform(X_test)
    </p>
    <p>
      final_predictions = final_model.predict(X_test_prepared)
    </p>
    <p>
      
    </p>
    <p>
      final_mse = mean_squared_error(y_test, final_predictions)
    </p>
    <p>
      final_rmse = np.sqrt(final_mse)
    </p>
    <p>
      
    </p>
    <p>
      print (&quot;final_predictions\n&quot;, final_predictions )
    </p>
    <p>
      print (&quot;final_rmse \n&quot;, final_rmse )
    </p>
    <p>
      
    </p>
    <p>
      confidence = 0.95
    </p>
    <p>
      squared_errors = (final_predictions - y_test) ** 2
    </p>
    <p>
      mean = squared_errors.mean()
    </p>
    <p>
      m = len(squared_errors)
    </p>
    <p>
      
    </p>
    <p>
      # from scipy import stats
    </p>
    <p>
      print(&quot;95% confidence interval: &quot;,
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;np.sqrt(stats.t.interval(confidence, m - 1,
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;loc=np.mean(squared_errors),
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;scale=stats.sem(squared_errors)))
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;)
    </p>
  </body>
</html></richcontent>
</node>
</node>
</map>
