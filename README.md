![NLP Header](https://github.com/boogiedev/UFO-SIGHTINGS/blob/master/media/ufo-header.png)

<p align="center">
  <img src="https://img.shields.io/badge/Maintained%3F-IN PROG-blue?style=flat-square"></img>
  <img src="https://img.shields.io/github/commit-activity/m/boogiedev/UFO-SIGHTINGS?style=flat-square"></img>
</p>


## Team

[Tyler Woods](https://github.com/tylerjwoods)  | [Joseph Shanks](https://github.com/josephshanks) | [Wesley Nguyen](https://github.com/boogiedev)
---|---|---|

 
## Table of Contents

- [Basic Overview](#basic-overview)
  - [Context](#context)
  - [Goal](#goal)
- [Exploring Data](#exploring-data)
  - [Initial Intake](#initial-intake)
  - [Feature Engineering](#feature-engineering)
- [Language Processing](#language-processing)
  - [Visualizations](#visualizations)
  - [Tokenizing](#tokenizing)
- [Future Considerations](#future-considerations)
- [License](#license)
- [Credits](#credits)
- [Thanks](#thanks)

## Basic Overview

### Context

<img align="right" src="https://i.pinimg.com/236x/32/47/16/324716a77ab7183025a1ad46786de375--x-files-funny-love-puns.jpg">

It's a bird... it's a plane...it's... a U.F.O. sighting? Over the course of human history U.F.O. sightings seem to be commonplace; commonly described as "flying saucers", strange lights and objects or straight up "Aliens". There are a lot of unknowns that surround the idea of unidentified flying objects, but one thing that is known, are that people are continually fascinated by them. Today, we are looking at reported U.F.O. sightings from ![THE NATIONAL UFO REPORTING CENTER](http://www.nuforc.org/). These are anonymous reports from various people all over the U.S. and sometimes even internationally.

### Goal

Using Natural Language Processing, we are hoping to parse through these sighting reports and explore possible commonalities, insights, and sentiment about these suspicious objects. By doing this, we are hoping to gain a more concrete truth of whether these sightings are figments of people's imaginations, or that there might be actually be an alien overlord visiting us from time to time.

> If reports from Alabama mention that this mysterious "object" has the shape of a rectangle, while reports from other states express the same thing, is it possible to make a connection here?




## Exploring Data

SOURCE             | TIMEFRAME 
:-------------------------:|:-------------------------:|
![NUFORC](http://www.nuforc.org/)  | MAY 10th 2017  

The specific data that we are focusing on today are U.F.O. reports from May 10th, 2017, with a total of 99 sightings. Below is a preview of the format that the data comes in from the website.

<p align="center">
  <img src="https://raw.githubusercontent.com/boogiedev/UFO-SIGHTINGS/master/media/dataexcerpt.PNG"></img>
</p>



### Initial Intake

Here is a detailed description of the intake data:
- `ID`: Report ID
- `url`: Report URL
- `html`: Raw HTML from report
- `time`: Date Time Object of the report

<p align="center">
  <img src="https://raw.githubusercontent.com/boogiedev/UFO-SIGHTINGS/master/media/dfbefore.PNG"></img>
</p>


### Feature Engineering / Data Cleaning

Looking at this data, we noticed that there were a couple things that could be cleaned and changed. 

```
def clean_data(data:pd.DataFrame) -> pd.DataFrame:
    """Cleaner for UFO DataFrame"""
    # Copy data to avoid collision
    df_copy = data.copy()
    
    
    # Rename ID Column for Clarity
    df_copy.columns.values[0] = 'ID'
    
    # Convert Time to DateTime Object
    df_copy['time'] = pd.to_datetime(df_copy['time'])
    
    # Parse data from HTML Column
    df_copy['state'] = None
    df_copy['content'] = None
    df_copy['shape'] = None
    for i in range(len(df_copy)):
        soup = BeautifulSoup(df_copy['html'][i], 'html.parser')
        meta_data = soup.find_all('tbody')[0].find_all('tr')[0]
        s = meta_data.get_text('|', strip=True).split("|")
        # store data into a dictionary
        s_dict = {x.partition(":")[0]:x.partition(":")[-1] for x in s}
        state = s_dict['Location'][-2:]
        df_copy.loc[i, 'state'] = state
        entry = soup.find_all('tbody')[0].find_all('tr')[1]
        df_copy.loc[i, 'content'] = entry.get_text(strip=True)
        duration = s_dict['Duration']
        df_copy.loc[i, 'duration'] = duration
        shape = s_dict['Shape']
        df_copy.loc[i, 'shape'] = shape
    
    return df_copy
```

We ended up parsing the data in the 'HTML' rows of our dataframe in order to extract useful information, using the cleaner function above we succesfully created new columns. The resule of this are below:

- `ID`: Report ID
- `url`: Report URL
- `html`: Raw HTML from report
- `time`: Date Time Object of the report
- `state`: Reported Location
- `content`: Content of the report
- `shape`: Alleged Shape of the UFO
- `duration`: Duration of the report sighting

<p align="center">
  <img src="https://raw.githubusercontent.com/boogiedev/UFO-SIGHTINGS/master/media/dfafter.PNG"></img>
</p>


---
## Language Processing

### Visualizations

Choropleth Map: 

> The choropleth map shown in the notebook 'choropleth_map.ipynb' shows the number of reports from each state in the time period, as well as the three most common words from those reports of each state.
With this information, we see a detailed image of where the reports are coming from and what the reports are talking about.
15:51

Observed UFO Shapes:

The bar chart shows what the most common shapes are in the reports. We can see that 'Circles' and 'Teardrops' are common shapes, as well as individuals reporting just seeing 'Light'.


<p align="center">
  <img src="https://github.com/boogiedev/UFO-SIGHTINGS/blob/master/media/observed_ufo_shapes.png?raw=true"></img>
</p>





### Tokenizing

After creating our stop words list and removing punctuations we tokenized our documents. “Tokenize” means creating “tokens” which are atomic units of the text. These tokens are words extracted by splitting the document.We then used the “SnowballStemmer” to stem our tokenized words. We decided to use the snowball stemmer over the WordNetLemmatizer or the PorterStemmer. The reason for this is show below. 

```python
## Print the stemmed and lemmatized words from the first document
print(“%16s | %16s | %16s | %16s |” % (“WORD”, “PORTER”, “SNOWBALL”, “LEMMATIZER”))
for i in range(min(len(docs_porter[0]), len(docs_snowball[0]), len(docs_wordnet[0]))):
  p, s, w = docs_porter[0][i], docs_snowball[0][i], docs_wordnet[0][i]
  if len(set((p, s, w))) != 1:
    print(“%16s | %16s | %16s | %16s |” % (doc_filter[0][i], p, s, w))
```
    
We chose to stem the words with the Snowball Stemmer due to its preservation of important words for this usecase such as ‘venus’The Snowball Stemmmer normalizes these words from its appeared form into their root form.We now have our list of clean tokens for each document! We turned this into a pandas Series to compute the TF-IDF

## Future Considerations

Using NaieveBayes to test comminalities of words used to derive if these occurences are related.


Do the U.F.O. sightings have a similar distribution of reports from states?


## License
[MIT ©](https://choosealicense.com/licenses/mit/)

## Credits

Fill

## Thanks

Fill
