
# coding: utf-8

# # Logistic Regression in Python

# Original Tutorial by [Greg from yhat](http://blog.yhat.com/posts/logistic-regression-python-rodeo.html)  
# Additional Notes by [onlyphantom](https://github.com/onlyphantom/)
# 
# Date: 7 May 2017

# **Logistic Regression** is a statistical technique capable of predicting a binary outcome and commonly applied in disciplines from credit and finance to medicine and other social sciences. 

# In[39]:

# Initialization (important!)
get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')

# Let's import the required packages
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

# Load the data using pandas.read_csv()
df = pd.read_csv("binary.csv")


# ## Summary Statistics & Exploratory Analysis

# In[23]:

# Let's inspect the df.head()
print(df.head())


# The dataset contains several columns which we use as predictor variables:
# - `gpa`  
# - `gre` score  
# - `rank` or prestige of an applicant's undergraduate alma matter  
# 
# The column `admit` is our binary target variable

# The column named `rank` could present a problem since `rank` is also the name of a method belonging to pandas `DataFrame`; Specifically, `rank` calculates the ordered rank (1 through n) of a `DataFrame/Series`. We want to rename our `rank` column to 'prestige'.

# In[24]:

df.columns = ["admit", "gre", "gpa", "prestige"]
print(df.columns)


# A `pandas` function analogous to `summary` in R is the `describe()` function. 

# In[6]:

df.describe()


# While the standard deviation is already included in the summary, we could specifically inspect it by calling `df.std()` too:

# In[7]:

df.std()


# Another feature similar to `table` in R is the `crosstab` function from `pandas`

# In[25]:

# multidimensional freq tables
pd.crosstab(df['admit'], df['prestige'], 
           rownames=['admit'])


# In[25]:

df.hist(color="black")
plt.show()


# ## Data preparation

# ### Dummy variables
# `pandas` gives us a great deal of control over how categorical variables are represented. Here we'll **dummify** the "prestige" column using `get_dummies`.
# 
# `get_dummies` creates a new `DataFrame` with binary indicator variables for each category / option in the column specified. In this case, `prestige` has four levels: 1 being most prestigious and 4 least. 
# 
# When we call `get_dummies` we get a dataframe with 4 columns of binary values (0 or 1) indicating which level the initial data point belongs to. 
# 

# In[26]:

dummy_ranks = pd.get_dummies(df['prestige'], prefix = 'prestige')
dummy_ranks.head()


# Create a clean data frame for our logistic regression model later:

# In[27]:

cols_to_keep = ['admit', 'gre', 'gpa']

# use .join to combine the columns 
# df[[ col2, col4 ]] allows us to subset columns 2 and 4
data = df[cols_to_keep].join(dummy_ranks[['prestige_2', 'prestige_3', 'prestige_4']])
data.head()


# Notice how we did not `prestige_1`, that is because the lack of any `1` between prestige 2 to 4 would indicate a level of `prestige_1`. When we treat `prestige_1` as our baseline and exclude it from our fit we also prevent [multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity#Remedies_for_multicollinearity), or the [dummy variable trap](https://en.wikipedia.org/wiki/Dummy_variable_(statistics) which is a result of including a dummy variable for every single category.

# In[28]:

# Manually add the intercept
data['intercept'] = 1.0
data.head()


# The intercept we added serve as a constant term for our Logistic Regression. The `statsmodels` function we use later will require that our intercepts / constants are specified explicitly.

# ## Performing the regression

# Recall that we are predicting the `admit` column using `gre`, `gpa` and the prestige dummy variables 2 through 4. 

# In[29]:

# subset 2nd column to last
train_cols = data.columns[1:]

logit = sm.Logit(data['admit'], data[train_cols])

# fit the model
result = logit.fit()


# ## Interpreting the result

# In[30]:

result.summary()


# Here we get an overview of the coefficients of the model, see how well these coefficients fit, the overall fit quality and other statistical measures. 
# 
# From the `coef` column, we can observe the **logistic regression coefficiencts**, which describes the change in the log odds of the outcome for a one unit increase in the predictor variable. 
# 
# - For every one unit change in `gre`, the log odds of admission (versus non-admission) increases by *0.0023*  
# - For a one unit increase in `gpa`, the log odds of admission increases by *0.8040*  
# - The indicator variables for `rank` are interpreted slightly differently. Here, having attended an undergraduate institution with a prestige of 2 versus one with a prestige of 1 will change the log odds of admission by *-0.675*  
# 
# The result object also allow us to isolate and inspect parts of the model output. The confidence interval gives us an idea for how robust the coefficients of the model are:

# In[33]:

result.conf_int()


# In the case here, we can be reasonably confident that there is an inverse relationship between the probability of being admitted and the prestige of a candidate's undergraduate school. 
# 
# > Notice that none of the confidence interval crosses the 0 point, indicating that these variables are all influential factors of deciding whether or not a student is accpeted. That said, `gre` has a weaker correlation than say, `prestige_4`.

# ### Odds ratio

# > It's useful at this stage to refresh our memory on what are **odds** and how they differ from **probability**? 
# 
# > Say the probability of success of some event is .8, then the rules of probability of failure is 1-.8 = .2. The odds of success are defined as the ratio of the probability of success over the probability of failure. Using the example above, the odd of success are .8/.2 = 4. We'd say that the odds of success are 4 to 1. If the probability of success is .5, then we have a 50-50 chance and the odds of success is 1 to 1. Logically enough, odds increase as the probability increases. 
# 
# > While probability ranges from 0 to 1, odds range from 0 to positive infinity. Building on that, the transformation from odds to log of odds is the log transformation and again, a monotonic transformation - that is to say, the greater the odds, the greater the log of odds. 
# 
# > A good reason of us tranforming probability to log odds in the first place is to model a variable with a range of 0 and 1, as in the case of probability. The transformation essentially gets around this restricted range by mapping probability ranging between 0 and 1, to log odds ranging from negative infinity to positive infinity. 
# 
# 
# If we take the exponential of each of the coefficients to generate ther odds ratiosm, this tells us how a 1 unit increase or decrease in a variable affects the odds of being admitted (instead of log of odds).
# 
# From what we've learnt in the model's output, we can expect the odds of being admitted to decrease by about 50% if the prestige of a school is 2. 

# In[34]:

# exp applies exponent on our result.params
np.exp(result.params)


# We can also perform the calculations using the coefficients estimated using the confidence interval to get a better picture of how uncertainty in variables can impact the admission rate. 

# In[35]:

# params: a vector of coefficients of each variable
params = result.params

# our 95% CI
conf = result.conf_int()

# Add a Odds Ration column
conf['OR'] = params

# Rename the columns
conf.columns = ['2.5%','95.5%', 'OR']

# Exp() on the conf table
np.exp(conf)


# ## Digging a little deeper

# As a way of evaluating our classifier, we'll recreate the dataset with every logical comination of input values. This will allow us to see how the predicted probability of admission increases / decreases across different variables. 
# 
# Before we do that we'll need to generate the combinations using a helper function called `cartesian`. We're also going to use `np.linspace` to create a range of linearly spaced values from a specified min and max value, in our ccase, just the min / max observed values.

# In[36]:

# instead of generating all possible values of GRE and 
# GPA, we will create a "linearly spaced" range of 10 values
# from the min to the max using np.linspace()

gres = np.linspace(data['gre'].min(), data['gre'].max(), 10)
print(gres)

gpas = np.linspace(data['gpa'].min(), data['gpa'].max(), 10)
print(gpas)

# define the cartesian function
def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

# enumerate all possibilities
combos = pd.DataFrame(cartesian([gres, gpas, 
                                 [1,2,3,4], [1.]]))

print(combos.head())


# In[37]:

# recreate the dummy variables
# map the column names to the respective columns
combos.columns = ['gre', 'gpa', 'prestige', 'intercept']
dummy_ranks = pd.get_dummies(combos['prestige'], 
                            prefix = 'prestige')
dummy_ranks.columns = ['prestige_1', 'prestige_2', 
                       'prestige_3', 'prestige_4']

# keep only what we need for making predictions
cols_to_keep = ['gre', 'gpa', 'prestige', 'intercept']
combos = combos[cols_to_keep].join(
    dummy_ranks.ix[:, 'prestige_2':])

# make predictions on the enumerated dataset
combos['admit_pred'] = result.predict(combos[train_cols])

print(combos.head())


# Now that we've generated our predictions, let's make sopme plots to visualize the results. We'll create a small helper function called `isolate_and_plot` which allow us to compare a given variable with the different prestige levels and the mean probability for that combination. 
# 
# To isolate prestige and the other variable I used a `pivot_tavle` which allows us to easily aggregate the data.

# In[41]:

def isolate_and_plot(variable):
    # isolate a specified variable and prestige rank
    grouped = pd.pivot_table(
        combos, values = ['admit_pred'], 
        index=[variable, 'prestige'], 
               aggfunc = np.mean)
    
    # make a plot
    colors = 'rbgyrbgy'
    for col in combos.prestige.unique():
        plt_data = grouped.ix[grouped.index.get_level_values(1)==col]
        pl.plot(plt_data.index.get_level_values(0), 
                 plt_data['admit_pred'], color=colors[int(col)])
        
    pl.xlabel(variable)
    pl.ylabel("P(admit=1)")
    pl.legend(['1', '2', '3', '4'], loc='upper left', 
                   title='Prestige')
    pl.title("Prob(admit=1) isolating "+ variable + 
                 " and prestige")
    pl.show()
        
isolate_and_plot('gre')
isolate_and_plot('gpa')


# The resulting plots show how `gre`. `gpa` and `prestige` affect the admission levels. We observe that the probability of admission gradually increases as gre and gpa increase and that the different prestige levels yield drastic probabilities of admission.

# ## Takeaways
# 
# Logistic regression is an excellent algorithm for regression. Even though some of the more advanced black box classification algorithms like SVM and RandomForest can perform better in some cases, there is value in knowing exactly what our model is doing. Often times, we can get by using RandomForest to select the features of our model and then rebuild the model with Logisitc Regression using the best features. 

# ## Credits
# 
# Credits to the original tutorial on [yHat](http://blog.yhat.com/posts/logistic-regression-python-rodeo.html), which is adapted from the R tutorial published by [UCLA](http://stats.idre.ucla.edu/r/dae/logit-regression/)
# 
# The code for our cartesian function originally found on StackOverflow [here](http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays)

# In[ ]:



