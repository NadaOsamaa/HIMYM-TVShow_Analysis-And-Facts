# "How I Met Your Mother" Analysis And Facts

This code analyzes the data and transcript from the TV show "How I Met Your Mother". The data includes information on the show's episodes, such as their titles, the season they belong to, directors, writers, ratings, and viewership, it also includes the transcript of the first 6 seasons of the show. 

The code extracts and explore the most common words used by each character, as well as their catchphrases, cleans the text data by removing punctuations and stop words, then generates bar charts and WordCloud images to visualize the analysis. In addition to that, it provides insights into various aspects of the show, including the number of seasons and episodes, distribution of episodes across seasons, and episode directors/writers. 


## Requirements

To run this code, you will need the following Python libraries:

- pandas
- numpy
- re
- nltk
- matplotlib
- plotly
- wordcloud

You can install these libraries using pip:

```
pip install pandas numpy re nltk matplotlib plotly wordcloud
```

You will also need to download the stopwords from the nltk library by running the following command in a Python console:

```
import nltk
nltk.download('stopwords')
```


## Data Cleaning

The data needed some cleaning before analysis, which was done using Pandas. Some of the cleaning steps performed include:

- Renaming columns
- Creating new columns by slicing from existing columns
- Cleaning and formatting column values
- Reordering columns
- Sorting values

## Data Analysis

The cleaned data was then used for various analyses. Some of the analyses performed include:

- Displaying the number of seasons and episodes
- Creating a pie chart showing the number of episodes per season
- Creating a sunburst chart showing the distribution of episodes directed by each director
- Creating a horizontal bar chart analyzing the average viewership and ratings for episodes directed by each director
- Creating a sunburst chart showing the distribution of episodes written by each writer
- Creating scatter plots analyzing the average viewership and ratings by season and episode.

The analysis results include:

- WordClouds for the most common words used by each character
- A figure displays the number of times each catchphrase was said by each character
- Interactive indicators for some words used by the character
- Season-wise statistics for views, ratings, directors, and writers


## Credits

- This code was created by [Nada Osama](https://github.com/NadaOsamaa)
- The data was obtained from Kaggle:
  - ["How I Met Your Mother" Dataset ↗](https://www.kaggle.com/datasets/vidheeshnacode/how-i-met-your-mother-himym-dataset)
  - ["How I Met Your Mother" Transcript ↗](https://www.kaggle.com/datasets/gibsonhu85/howimetyourmotherscript)
