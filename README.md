# Private
Private GitHub repo

Pretty sure we dont need to be doing one hot encoding (even thought im doing it still) or normalizing (minmaxscale) since were using tree methods.

Considerations & Notes:

- Would be nice to be able to use like 4000 n_estimators; model runs for too long if I try that. So we are at 100.

- feature selection (keep 95% variance) or maybe PCA???

- lightgbm (its a microsoft thing so its a hassle to implement on macos)

- k fold cross validation?

- merging tables (this goes along with EDA)

some ideas:
its not quite clear what these two people are doing for null values in their code, other than that theres some ideas
https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm/code - lightgbm mode
https://www.kaggle.com/shivamb/homecreditrisk-extensive-eda-baseline-model/notebook -lightgbm model

https://www.kaggle.com/eliotbarr/stacking-test-sklearn-xgboost-catboost-lightgbm/code - stacked models for this competition

https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python - model stacking tutorial
