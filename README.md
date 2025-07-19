
# Inclusive Scripts: Predicting Media Diversity and Bechdel Test Scores

Jhanvi Udani

---

## Disclaimer

The research conducted is based on publicly available data and information and was used solely for academic purposes. The authors have taken utmost care in analyzing and using the data, to ensure no biases are involved and appropriate conclusions are drawn. They also ackowledge that gender diversity, media diversity are sensitive topics that exists on a spectrum and there is no single method or formula to tackle these issues. All work is the original work of the authors and due credit is given for any material that was referenced.

## Problem Description and Context

The underrepresentation of diverse characters and narratives in movies is a longstanding issue. Scriptwriters often lack tools to assess the diversity of their work during the writing process. This project aims to develop a machine learning model that predicts both Bechdel Test scores and a broader Diversity Score for movies. By analyzing scripts and metadata, the model will extract features that contribute to these scores, empowering scriptwriters to improve representation in their work. We were motivated to work on this topic becomes it is at the intersection of work related to Diversity and Representation, and very interesting technological tools we all wanted to learn.

## Ethical Considerations

The main ethical considerations we keep in mind is keeping the data representative and not losing too much information. We chose to keep our data in seperate data sets so that we don't lose significant data while merging our datasets, we keep this in mind while performing any analysis. 

Additionally, our Script and Movie Dataset also contains data from various countries that have English (or dubbed in English) movies and not just the US, this helps us derive unbiased insights from many countries. 

We also acknowledge that it might be tempting to manipulate the data to get the results we wish to see, thus to avoid that we are using all data as derived from the source. We want our results to be accurate even if they are not perfect or what we would like to see. 

## Datasets

**1. Bechdel Test API**

* Source: [https://bechdeltest.com/api/v1/doc](https://bechdeltest.com/api/v1/doc)
* Format: JSON
* Variables: Movie title, Bechdel Test score (Pass/Fail)
* Description: Provides Bechdel Test scores for over 8,000 movies.

**2. Mediaversity Index**

* Source: [https://docs.google.com/spreadsheets/d/1-vqRNK_F0_D3V-e1eeUcZVhDBqg9ogixENbVPNoiSz8/edit#gid=1991719371](https://docs.google.com/spreadsheets/d/1-vqRNK_F0_D3V-e1eeUcZVhDBqg9ogixENbVPNoiSz8/edit#gid=1991719371)
* Format: CSV
* Variables: Year, Film, Scores (Delineated by Technical, Gender, Race)
* Description: Offers diversity scores for 350+ movies which will help use in building a train set. 

**3. Movie-Script-Database**

* Source: [https://github.com/Aveek-Saha/Movie-Script-Database](https://github.com/Aveek-Saha/Movie-Script-Database)
* Format: Text files containing movie scripts
* Description: Collection of movie scripts for feature extraction.

**4. TMDb API**

* Source: https://developer.themoviedb.org/reference/intro/getting-started 
* Format: API that returns JSON
* Variables: Movie cast, crew, genres, ratings, etc.
* Description: Provides movie metadata for enriching features.

We have combined all the data compiled from these datasets into 6 main Excel files with the clean version of all this data and scripts that have been seperated into scenes and dialogues.

## Questions to Answer

1. **Target Variable:** Bechdel Test Score (Classification)
* **Question:**  What percentage of movies in the dataset pass the Bechdel Test? Can we identify patterns in scripts that distinguish Pass from Fail movies? Can we achieve comparable accuracy in predicting Bechdel Test scores using decision trees compared to if-else statements, considering computational efficiency? How does the frequency of scenes with two women characters conversing affect the Bechdel Test score?

2. **Target Variable:** Diversity Score (Regression)
* **Question:**  How do the number of characters from different races and the distribution of dialogue lines across genders and races contribute to the overall diversity score provided by the Mediaversity Index?

3. **Target Variable:** Features For Both 
* **Question:**  How can we use unsupervised learning techniques to better understand scripts - and better understand the difference between scenes that might pass the Bechdel test, and those that do not seem to pass it?

## Model Evaluations

**1. Classification to Identify Bechdel Scores**

To gain robust performance metrics for our problem, we developed a large dataset. This was done by running this Python package: https://github.com/Aveek-Saha/Movie-Script-Database, and using it to generate 1000s of scripts. We then matched on Bechdel available movies, and used that data to generate features. We used metrics related to Precision, especially focused on Group 3 (Pass Bechdel). This work can be seen in 1_Bechdel.

We are using classification because Bechdel is a pass/fail metric. As such, we are focused on the standard methods for classification - Random Forest, KNN, and Gaussian NB.

**2. Regression for MediaDiversity**

Further, to answer our second question of predicting the Media Diversity Scores using different features we used - LinearRegression, Random Forest, Gradient Boosting, Ridge, Lasso, KNN and SGD. This can be found in 2_Media.ipynb. We used regression to predict the scores and MSE as the most commonly used performance evaluation metric. A high MSE indicates poor model performance and a low MSE indicates good model performance. The Linear Regression Model has the lowest Test MSE while the Gradient Boosting Regressor has the lowest Train MSE. Both models show signs of overfitting. We also saw overfitting in the intial models to overcome which we tuned our parameters. We did not achieve much information from running these models. 

**3. Unsupervised Learning - Media Diversity and NLP**

For our third question, we have aimed at visualising the clusters for various features. We used NLTK to clean up the scripts and get clusters which can be seen in 3_Unsupervised.ipynb and used PCA, DBScan, Agglomerative Clustering and KMeans Clustering to visualize the clusters. However, in most cases we didn't see the formation of any clear clusters and will be refining the work further. We see that fewer components explain a large amount of the data and we would consider modelling further using fewer components. The main metric we are using and will continue using to judge the model performance is the Silhoutte Score. 

## Additional Models for Sprint 6

**Sentiment Analysis**
We added a Sentiment Analysis Model to see the different sentiments in the scripts and to compare the sentiments of the movies that pass the Bechdel Test and don't pass the Bechdel Test. This work can be found in the 3_SentimentAnalysis.ipynb. It uses the SentimentIntensityAnalyzer from NLTK to get the scores for negative, positive or neutral sentiments in the scripts.  We chose this as it is good with shorter content and we tried calculating the sentiments per line break. 

**Cluster Analysis using TFIDF**
We used TFIDF to rank scene "importance" in the 3_TFIDF.ipynb notebook. This was done by using Cosine scores to see how close scenes were to a movie's important words. After that, we clustered on that versus gender ratios to identify interesting patterns.

**Latent Dirichlet Allocation** 
We used LDA to identify the most important topics per types of Bechdel scenes, and were able to identify different genres between the two types of scenes. This model can be found in 3_TFIDF.ipynb notebook. 

**Market Basket Analysis**
Our group attempted to identify antecedents and consequents in non-Bechdel vs Bechdel scenes, this can be found in 3_MBA.ipynb. We were unable to identify anything super valuable. Unclear association between words for scenes that pass the bechdel test and those that don't was seen, and the next step would be to run a similar analysis on movies, instead of scenes. 

## Running the Project

For convenience and clarity, we compiled all the datasets obtained from the initial 4 sprints into the Dataset file on our repo. Our main analysis and the models for the questions is contained in 3 files - 1_Bechdel.ipynb, 2_Media.ipynb and 3_Unsupervised.ipynb. We have additional models under 3_MBA, 3_SentimentAnalysis and 3_TFIDF.

To look at our analysis for our first question -  Run 1_Bechdel.ipynb.

For our second question, the analysis is under 2_Media.ipynb

Finally, for our third question, run the file titled 3_Unsupervised. We also have additional models under 3_MBA, 3_SentimentAnalysis and 3_TFIDF to be run in the order they are mentioned.

To understand and see how we got our data in the dataset folder. You may look under Old_Files. This creates the Dataset folder. The file creation process is quite long and tedious, so we recommend taking out ample time before running any of these files.  

 1. To see how we converted script pdfs into legible TXT files, look at scriptCleaner.ipynb. A lot of that work is taken from Aveek-Saha's Movie Script Database.
 2. To see how we converted script TXT files into DataFrames, look at UnderstandingScripts.ipynb. 
 3. To see how we matched those characters with the TMDB database, sprint4/bechdel/ScriptForBechdel.ipynb shows that work.
 4. To see how we created DataFrames with all of the individual scenes, sprint4/bechdel/ScenesForBechdel.ipynb shows that work.
 5. To see a lot of this work in one place, look at MediaDiversity_Initial.ipynb.

## Team Contract

**1. List the names of all your teammates:**
Jhanvi Udani, Shambhavi Bhushan, Raj Shah

**2. Agree as a team, what branching strategy do you plan to use in your final projects? Justify your choices**
We are going to use Git Flow - Jhanvi and Raj have experience using that flow before, and think it will be easier for working through any conflicts and making sure our production code is safer.

**3.Communication: Outline how the team will communicate — including frequency and methods (e.g., email, WhatsApp, team meetings).  What is the maximum expected response time?**
The team will communicate daily over WhatsApp (on Weekdays) and make sure we respond within 12 hours to questions/concerns. We also will see each other on Tuesdays at 12:30 for weekly meetings.

**4. Decision-Making: How will decisions be made in this team? How will you stay on track? Do you plan on having meetings or any strategies for working through your final project**
Decisions will be made through conversations - larger decisions in person, and smaller ones through WhatsApp. We will stay on track by doing a very light agile process.

**5. As with any team project there is always the possibility of conflict arising, if it does in the future, how will you resolve differences? List at least two strategies**
Strategy One: We will listen to everyone's pespective before resorting to majority rules.
Strategy Two: For major conflicts we will seek help from TAs and Prof. GS 

**6.Commitments: How will you handle different levels of participation and commitment? What process will you follow if someone does not live up to his/her/their responsibilities? (3-5 sentences)**
We will divide all work equally and have internal deadlines in place, accomodating different schedules. If someone needs more time or help finishing as long as they communicate it before the internal deadline we will be happy. If they don't on multiple occassions, we will seek help. 

**7.Diversity: How will you accommodate different learning and working styles? Talk about your own styles and schedules for working and come to an agreement (3-6 sentences)**
We all learn by doing so we will divide responsibilites equally so that everyone has the opportunity to learn. Shambhavi doesn't have coding experience so Raj and Jhanvi will help her. We will respect each other's busy schedules. 


### Dataset Notes

1. **Cleaning:** Because a lot of the data comes from APIs, there wasn't too much cleaning to be done on the Movie ID side, 
except manually adding in some movie data where it was missing. However, the Script Data requires lots of preprocessing and cleaning to ensure features can be grabbed from the Scripts.
2. **Merging:** We did not merge all our datasets but cleaned it to build seperate datasets used in all of our models. This helped us retain maximum sscript data and features. 
3. **Visualizations** The files `UnderstandingScripts` and `Cast` contain visualizations related to the project. `Cast` shows
some gender data, while `UnderstandingScripts` shows some data relating to actual lines in a film.

### References

* https://en.wikipedia.org/wiki/Bechdel_test 
* https://bechdeltest.com/
* https://github.com/Aveek-Saha/Movie-Script-Database
* https://developer.themoviedb.org/reference/movie-details
* https://www.themoviedb.org/talk/6422c9fc8d22fc00a9df02b7
* https://developer.themoviedb.org/reference/person-details
* Python 2: Lecture Slides- 9-web+data
* https://developer.themoviedb.org/reference/search-movie
* https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.at.html
* https://bechdeltest.com/
* https://github.com/Aveek-Saha/Movie-Script-Database
* https://developer.themoviedb.org/reference/movie-details
* https://www.themoviedb.org/talk/6422c9fc8d22fc00a9df02b7
* https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.at.html
* https://www.geeksforgeeks.org/stochastic-gradient-descent-regressor/
* https://www.analyticsvidhya.com/blog/2021/10/support-vector-machinessvm-a-complete-guide-for-beginners/ 
* More references are mentioned and credited in our Jupyter notebooks

### Future Work

* Explore additional data sources like casting information or character descriptions.
* Add additinal features to feature engineering
* Investigate alternative models for Diversity Score prediction 
