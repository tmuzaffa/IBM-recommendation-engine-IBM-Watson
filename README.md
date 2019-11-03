# IBM-recommendation-engine-IBM-Watson
IBM recommendation-engine for artciles


### Table of Contents

1. [Libraries used for the project](#libraries)
2. [Objective](#motivation)
3. [File Descriptions](#files)
4. [Analysis](#Analysis)
5. [Charts](#Charts)
6. [Acknowledgements](#acknowledgements)

## Libraries used for the project <a name="libraries"></a>

Following python libraries:

1. Pandas
2. Matplotlib 
3. NLTK
4. NumPy
5. Pickle
6. Seaborn
7. Sklearn
8. accuracy_score


I used the Anconda python distribution with python 3.0

## Objective<a name="motivation"></a>

The objective of this project is to build a model and classify messages during a disaster. We have been given disaster twitter messages data set which have 36 pre-defined categories. With the help of the model, we can classify the message to these categories and send the message to the appropriate disaster relief agency. For example, we do not want Medical Help message to food agency as they wont be able to help the person in time. 

This project will involve building an ETL pipeline and Machine Learning pipeline. Objective of this task is also multiclassification. We want one message to be classified to multiple categories if needed. 

This data set is  provided to us by IBM
## File Descriptions <a name="files"></a>




data:
- disaster_message.csv
- disaster_Categories.csv


Recommendation Engine:
- train_classifer.py
- ML Pipeline Preparation.ipynb


app:
- go.html





## Analysis<a name="Analysis"></a>

- By looking at the results of .describe(), we can see that the article_ids of both data frames  do not match with each other, df data frame has article id range from 0 to 1444, where as  df_content range from 0 to 1444.
- The number of unique articles that have at least one interaction are 714
- The number of unique articles on the IBM platform are 1051
- The number of unique users are 5148
- The number of user-article interactions are 45993
- Since we have no information about the new user, so collaborative filtering will not be the best option, as we will not be able to compare and match with other users and provide recommendations, however we can use top rated recommendations, and just recommend top rated articles to new users, which might not be the best option as well, as new user might have already read those artilces. I think to deal with such a problem, we should use FunkSVD

- As we have seen above that classes are highly imabalnced as there are lots of articles that user have no interacted with, as shown by the large proportion of class 0 as comapred to class 1. There might be some overfitting, as we have used 740 latent features, which makes the model complex and takes into account even small changes, however we can rectify that by making the model simpler and using less latent features. We can see above that the best accuracy of test, train sets can be achieved when we use approximately 100 latent features. If we use above, it leads to over fitting.

Since we can only predict for 20 users, so above fitting might not be accurate representaiton for our prediction. A better metric will be the article recommendation and not article interaction. It could be that user never interacted with the article becuase they donot like the it, or may be user likes the articles but have no just interacted with it yet on the platform.

We can also use A/B testing to see how our predictions are working. We can assign half users to control group and half to experimental group.



## Charts<a name = "Charts"></a>
![Histogram of user ineraction with articles](https://github.com/tmuzaffa/Disaster-Response-Pipeline-prediction/blob/master/plotly2.JPG)
![Accuracy vs. number of latent features](https://github.com/tmuzaffa/Disaster-Response-Pipeline-prediction/blob/master/plotly2.JPG)
![Accuracy vs number of latent features](https://github.com/tmuzaffa/Disaster-Response-Pipeline-prediction/blob/master/plotly2.JPG)

  
  
## Acknowledgements<a name="acknowledgements"></a>

- This data set is  provided to us by IBM
- .dsecribe() > https://www.geeksforgeeks.org/python-pandas-dataframe-describe-method/
- Pyplot> https://matplotlib.org/3.1.1/gallery/pyplots/pyplot_text.html#sphx-glr-gallery-pyplots-pyplot-text-py

- Remove/find duplicates > https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.duplicated.html

- Nunique() > https://www.geeksforgeeks.org/python-pandas-dataframe-nunique/
- Most frequent index > ### Ref: https://stackoverflow.com/questions/48590268/pandas-get-the-most-frequent-values-of-a-column/48590361
- Unstack data frame > #Ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.unstack.html
- Setdiff1d  > https://docs.scipy.org/doc/numpy/reference/generated/numpy.setdiff1d.html
- Linear Algebra > Ref: https://docs.scipy.org/doc/numpy/reference/routines.linalg.html


