# store_sales_predictor
There are two jupyter notebooks and a python script file.
> store_sales_predictor.ipynb

This file does the data analysis of train.csv and make several GLM models but they had low scores
Due to this reason I had to train a 3 layer NN, but due to an issue jupyter notebook hung up while traing
so I had to train the model on terminal in the file

> store_sales_predictor.py

In this file I trained the NN and saved the model

> predict.ipynb
This file loads the model and test dataset and predict the results which are stored in **ans.csv**
