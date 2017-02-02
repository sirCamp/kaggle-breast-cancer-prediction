# Different Approaches to predict malignous breast cancers based on Kaggle dataset
This project is started with the goal use machine learning algorithms and learn how to optimize the tuning params and also and hopefully to help some diagnoses.

These are different approaches like:
 + ANN
 + DecisionTree
 + Bayes 
 + KNeighbors
 
Mentioned as the goal of the project is to predict the right way if there are a breast cancer or not.

The whole project is written in Python.
All the parameters of algorithms are tuned at best possible and the Reached accuracy is around ~ 94%.
To be precise the lower Obtained result is around 90% and the best is over 97% with 94% as mean.

This different results are caused by the shuffling of the elements. That is Necessary to make the data more "reals".

Each algorithm work trainset on 70% of initial dataset and it is tested with the 30%.


###How to
To run the scripts you just type:
```python
python script_name.py
```
As result of execution the reached accuracy will print

* the dataset can be found [here](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)