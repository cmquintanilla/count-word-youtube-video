"""
Credits for all the help to this website: https://www.cienciadedatos.net/documentos/py25-text-mining-python.html
"""
#Adding all the libraries needed
import pandas as pd
from pandas.io import json
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
import re
#These two lines are required to download the package of stopwords
#import nltk 
#nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt

#get the transcript for a list of Youtube videos using its ID
videoIds = ['1vOiEBcX9JI', 'hgYLOEwZmSU', 'xEG6mJ0iBTw', 'lMoZcbpfQR8', 'HODpgiOcYrY', 'oxXbS5fnkJ0', 'Fmlorlx91_Q', 'GQMs0_ACSyQ', '3NW438nFZL4', 'lBGzBPpn7yU', 'uBoY0NCdKCI', 'X0iZXF4futU', 'g_n3sUTDxkc', '_zKigryQc6U', 'YuXZIZSYz6Y', 'WUNshEcRsWY', 'OCpGfYo-neE', 'ra_1ZhQ64DQ']
transcripts = []
for videoId in videoIds:
    transcriptList = YouTubeTranscriptApi.list_transcripts(videoId)
    transcript = transcriptList.find_generated_transcript(['es'])
    transcript = transcript.fetch()
    transcripts.append(transcript)

#Formatting as Json
JSONTranscripts = []
for transcript in transcripts: 
    jsonFormatted = JSONFormatter().format_transcript(transcript)
    JSONFormat = json.loads(jsonFormatted)
    JSONTranscripts.append(JSONFormat)
    
#Creating a DF based on the JSONs
dfTranscripts = []
index = 0
for JSONTranscript in JSONTranscripts:
    index = index + 1
    df = pd.DataFrame.from_dict(JSONTranscript)
    df['class_number'] = index
    dfTranscripts.append(df)

#Concatening all the DF
dfWords = pd.concat(dfTranscripts)

#Function to cleaning and tokenizing the text
def cleaningAndTokenizer(text):
    #lower case
    newText = text.lower()
    #Delete punctuation marks
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    newText = re.sub(regex, ' ', newText)
    #Delete numbers
    newText = re.sub("\d+", ' ', newText)
    #Deleting multiple blank spaces
    newText = re.sub("\\s+", ' ', newText)
    # Individual Word Tokenizer
    newText = newText.split(sep = ' ')
    # Deleting tokens with len < 2
    newText = [token for token in newText if len(token) > 1]
    return newText
#Cleaning and tokenizing the text
dfWords['tokenized_text'] = dfWords['text'].apply(lambda x: cleaningAndTokenizer(x))

#Get rid of the unnecessary columns
words = dfWords.explode(column='tokenized_text')
words = words.drop(columns={'text', 'start', 'duration'})
words = words.rename(columns={'tokenized_text': 'token'})

#Deleting all the stopwords
stopWords = list(stopwords.words('spanish'))
words = words[~(words["token"].isin(stopWords))]

#Counting words
wordCount = words.groupby(by='token')['class_number'].count()
wordCount = wordCount.sort_values()
#printing top 10 most repeated words
print(wordCount.tail(10))

topTenWords = wordCount.tail(10)
#generating a bar chart for the top 10 most repeated words
labels = topTenWords.index.tolist()
y = np.array(topTenWords.values.tolist())
x = np.arange(len(labels))
width = 0.15
fig, ax = plt.subplots()
x1 = ax.bar(x - width/2, y, width, label='Words')

ax.set_ylabel('Count')
ax.set_title('Most spoken words in the Data Science Classes')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.bar_label(x1, padding=3)
fig.tight_layout()
plt.show()
