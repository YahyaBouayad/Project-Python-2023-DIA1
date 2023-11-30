# Drug consumption:

### Presentation of the raw data:

We chose the dataset on the study done on the drug available on this link:

[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified)

This one is composed of 5 rows × 31 columns, the data are mapped by values composed between -2 and 2 according to the deviation and variety of values. The column names are not present, but everything is available on the site where we were able to recover the dataset. The columns are composed of data on the person such as age, sex, education, but also data on the person's character, open-mindedness, impulsiveness ... And finally there are the data on several drugs consumed and at what frequency.


### Cleaning the data:

For the cleaning of the dataset we decided to remap the values with their true meaning, this will allow us to have a more interesting display to analyze and visualize.

You can see the new cleaned dataset, we also created lists to facilitate the management of the data we want to study.

Before starting the study we can see the correlation between all the data, this can give us ideas on the future data we will want to study.


### Data visualization:

We decided first to display simple data such as the distribution of age, sex, ethnicity, countries.

From this information we can already determine several pieces of information.

- The age distribution is very well distributed.
- There are as many women as men.
- We can notice that the "white" category for ethnicity and the "Uk" and "USA" countries is not very balanced (more than 90% are "white"). We can then assume that the initial data collection was mainly carried out in the US and UK which skews our future study a bit. We will therefore avoid taking these information into account for the rest of the study.

We also decided to display certain consumption in relation to certain characteristics.

Such as age and the consumption of a certain drug. Where we can notice that certain drugs are particularly consumed by young people, while others are consumed by all age groups.

We also displayed correlation matrices between certain types of information. Such as drug consumption and others. We can notice some interesting correlations such as the consumption of crack with the consumption of ecstasy. Or on certain characters, we notice a great correlation between sensation seeking and impulsiveness. This allows us to already make some hypotheses.

Finally, we were able to make a visual of the number of consumers in % for each drug.

This gives us ideas of analysis to do. We can for example notice that the majority wins on the consumption of coffee, chocolate or nicotine. But we can also notice that a majority has already tried Cannabis, this is perhaps due to the fact that some have already been able to try at least once in their life. The other drugs considered as hard have a very relevant user rate.

### Data Analysis:

Here we have done more thorough analysis where we can determine the effect of certain drugs.

We started by analyzing the characteristic of each category in relation to the drug studied. We can compare this graph with a "non-consumer" of the same drug. For example, if we choose cannabis as a drug, we can compare the average character of consumers against those who do not consume it.

We then obtain two "Radar" graphs ideal for visualizing several variables.

We can notice that for whatever drug given in entry the consumer and non-consumer end up having opposing characters. For powerful and addictive drugs, we find a lot of negative character that comes out.

The second graph represents the consumption trends by age category.

Here we can visualize some coherences that we can find for example in cannabis where here we can see a very high consumption at young age then which dissipates little by little.

We can notice that other drugs such as cocaine which are not necessarily accessible to students are particularly found in the age range 34-64.

This graph represents the distribution of education levels according to drug consumption. This graph confirms theories put forward. We can also notice that for certain very addictive and very dependent drugs the level of study is low.

This graph represents the frequency of the most present drug combinations. This can then confirm certain correlations that we were able to make at the beginning of the study. We can see here that Benzodiazepine is very present with cannabis with a daily addiction rate, followed by amphetamine and cannabis.

The last graph joins the first one but we can more easily compare the traits of each drug.

### Machine learning:

As in all the project we have coded in an encapsulated way to easily reuse our code, it is the same for machine learning. The majority of our functions are reusable for each model and each chosen X and Y.

For this part we decided to try to use machine learning to answer this question: "Can we determine how often a person is likely to consume a drug according to their characteristics and information?"

For this we have separated the dataset into training 80%/20% and to start we have used the SVC because very good for the Management of High Dimensions.

As one can imagine the result is not at the rendezvous from the start.

We then tried to use other models that could better adapt to our data such as the DecisionTreeClassifier (for its **Non-linearity** indeed it can capture non-linear relationships between features) and the RandomForestClassifier (for precision) but also the Knn but the latter did not succeed.

Here are our results:

As expected, the SVC remains the most efficient for our dataset.

We therefore tried the use of Hyper-parameter which will allow us to refine our model. For this we used the Hyper-parameters of the SVC, we judged that it is the best model for our study. This one is composed of several parameters such as the "C" which represents the regularization parameter, "gamma" the kernel coefficient and finally "Kernel" which represents the type of kernel. There are other parameters, but the more we add the longer our code will be for not necessarily a good result.

To apply the Hyper-parameter we used the GridSearchCV algorithm which is an exhaustive search technique used in machine learning to select the best set of parameters, it tests all possible combinations of specified hyperparameters in a grid. This allows you to find the combination that gives the best performance for a given model on a specific dataset.

After several applications this does not result in better precision.

After reflection we concluded that the dataset did not have enough data for an output of 7 possibilities. We therefore decided to add two main things. The first is the use of almost all the columns in input, this will allow us to have new data for the algorithm that will be able to find links between the drugs that the person has already consumed to find a link with the drugs that we give him to study. We also reduced the response to whether or not the person has consumed or will ever consume this drug. This reduces the research of the model.

So here is the modification made on the dataset:

We reused the code to visualize all the given models.

We then obtain a precision of almost 90% for the SVM.
