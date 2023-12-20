# Final-Project
### Introduction 

Race and gender are two of the most prominent and important aspects of identities in society. As most people internal biases both consciously and unconsciously, both race and gender affect how people are treated and perceived in the world, including in politics. I wanted to explore the intersection of these identities in politics, and my research question is “Does the public react to and comment on YouTube videos of people of color (POC) and white Congresswomen differently?”

My hypothesis is that comments on YouTube videos of POC congresswomen will contain more misogynistic comments and gendered microaggressions than that of non-POC congresswomen. Previous research has shown that women of color, particularly Black women, tend to be judged more harshly than their white and male counterparts. Their actions and behaviors are often deemed “emotional” and “angry” in situations in which a white man’s actions would not be. 

There were a number of historical restrictions to women and people of color holding office, but even presently women and POC make up an unrepresentative small proportion of Congress. One factor making it more difficult for women and women of color to successfully run for office may be the misogyny and racist misogyny they face on the campaign trail and in the job. This project works to gauge public perception and treatment of women in Congress by looking at YouTube comments. It also explores whether racial factors amplify the misogyny Congresswomen face. This project works to draw attention to the social biases regarding race and gender and how it affects politics. 

### Methods

Firstly, I used data from Rutgers’ Center for American Women and Politics. They keep and update a database of women elected officials and information such as their position, their district, and their race. Using the sample function in Python, I took a random sample of 5 Congresswomen who were white and 5 Congresswomen who were not white. 

Next I found the first policy-related speech listed when the name of the Congresswoman is searched on YouTube. To try and control for misogyny directed at policies and the content of the video, rather than the policymaker delivering the information, I excluded speech on heavily gendered issues, such as Grace Meng’s video on menstrual equity. Instead, I used the second video. Another issue I encountered was that some of the videos had the comments turned off, and I used the second video listed instead. 

Then in my code, I created a list of video IDs for the videos of white Congresswomen, and a separate list for the video IDs of POC Congresswomen. Next, retrieved my YouTube API Key through Google Cloud Console projects. Using resources from GeeksforGeeks, I created a function that retrieved comments from YouTube videos, which included replies to comments and from all pages of comments, and then stored the comments and replies in a list. I ran the list of video IDs through a for loop that applied the new function to each video ID and stored them. 

For data cleaning, first I imported the necessary libraries and tokenized the comments so I could use tools such as NLTK. I converted all the comments into lowercase and lemmatized the words, which help the term frequency and count frequency tools later identify terms that have different capitalization or are different parts of speech, but have the same meaning, as the same term. Additionally, I removed stop words using the english stopwords list in NLTK. Removing stop words is especially important when conducting term frequency-inverse document frequency (TF-IDF), as stop words are likely to appear many times and have high TF-IDFs, but are not important to subjects I am studying and analyzing. 

For my analysis, I ran CountVectorizer and TFIDFVectorizer to create a count and TF-IDF scores for each term, using resources from Mukesh Chaudhary's TF-IDF Tutorial. I had created a list of words that I believed have social associations and different use rates between genders and POC and non-POC. These words are angry, rude, emotional, bossy, friendly, aggressive, and loud. I then retrieved the TF-IDF scores for these terms across the comments on white Congresswomens’ videos and POC Congresswomen’s videos. As women of color, particularly Black women, tend to be judged more harshly than their white and male counterparts, I repeated this process for words with broader negative racial stereotypes, which contain lazy and stupid. 

Additionally, I looked at the terms with the highest TF-IDF scores to see if there were any patterns or interesting terms that stuck out to me, but they were mostly objects that were likely the topic of the videos. Some examples include people, China, mask, and American. 

### Results
Please also find the full code in the file in this repository named "Text_Analysis_Final_Project"!
```
import pandas as pd
from google.colab import files
uploaded = files.upload()
import io
legislators = pd.read_csv(io.BytesIO(uploaded['officeholders_positions-1702871133.csv']))
#Sampling
whitesample = legislators[legislators["race_ethnicity"] == "White"].sample(5)
pocsample = legislators[legislators["race_ethnicity"] != "White"].sample(5)
whitesample, pocsample
```
```
from googleapiclient.discovery import build

api_key = "AIzaSyB2WGUKxBl9fi-6E1dV1_237KTlMPWNaWg"

def video_comments(video_id):
    comments_and_replies = []
    youtube = build('youtube', 'v3', developerKey=api_key)

    video_response = youtube.commentThreads().list(
        part='snippet,replies',
        videoId=video_id
    ).execute()

    while video_response:

        for item in video_response['items']:

            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']

            replycount = item['snippet']['totalReplyCount']

            if replycount > 0:

                for reply in item['replies']['comments']:

                    reply_text = reply['snippet']['textDisplay']

                    comments_and_replies.append(reply_text)
            else:

                comments_and_replies.append(comment)

        if 'nextPageToken' in video_response:
            video_response = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                pageToken=video_response['nextPageToken']
            ).execute()
        else:
            break

    return comments_and_replies
```
```
white_video_ids = ["yvAg6OhMFyU", 'peT7U0ziOxs','GD2qRpGg6io','METLv1ou3c8','fXv4z2E6izk']

all_comments_and_replies = []
for x in white_video_ids:
    all_comments_and_replies.extend(video_comments(x))

poc_video_ids = ['ZIW1ECi3ZLE','AcimU7xwpRc','Ze0jW_ysAJ0','kML6_m4ntvU','1DpivHkMd9s']

poc_comments_and_replies = []
for x in poc_video_ids:
    poc_comments_and_replies.extend(video_comments(x))

print(len(poc_comments_and_replies))
print(len(all_comments_and_replies))
```
```
#data cleaning
import nltk
import matplotlib
%matplotlib inline
import json
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.corpus import stopwords
whitecomment = [word_tokenize(comment) for comment in all_comments_and_replies]
poccomment = [word_tokenize(comment) for comment in poc_comments_and_replies]

poclist = [word for comment in poccomment for word in comment]
whitelist = [word for comment in whitecomment for word in comment]
```
```
poclist = [t.lower() for t in poclist if t.isalpha()]
whitelist = [t.lower() for t in whitelist if t.isalpha()]
nltk.download('stopwords')
stops = stopwords.words('english')
nltk.download('averaged_perceptron_tagger')
poclist = [t for t in poclist if t not in stops]
whitelist = [t for t in whitelist if t not in stops]
```
```
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

from nltk.corpus import wordnet

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
nltk.download('wordnet')
poclist = [wordnet_lemmatizer.lemmatize(t, get_wordnet_pos(t)) for t in poclist]
whitelist = [wordnet_lemmatizer.lemmatize(t, get_wordnet_pos(t)) for t in whitelist]
```
```
separator = ' '
pocdoc = separator.join(poclist)
whitedoc = separator.join(whitelist)
```
```
!pip install sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
pd.options.display.max_rows = 600
from pathlib import Path
import glob
```
```
both = [pocdoc,whitedoc]
```
```
countvectorizer = CountVectorizer(analyzer= 'word', stop_words='english')
tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')
count_wm = countvectorizer.fit_transform(both)
tfidf_wm = tfidfvectorizer.fit_transform(both)
count_tokens = countvectorizer.get_feature_names_out()
tfidf_tokens = tfidfvectorizer.get_feature_names_out()
df_countvect = pd.DataFrame(data = count_wm.toarray(),index = ['Doc1','Doc2'],columns = count_tokens)
df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),index = ['Doc1','Doc2'],columns = tfidf_tokens)
print("Count Vectorizer\n")
print(df_countvect)
print("\nTD-IDF Vectorizer\n")
print(df_tfidfvect)
```
```
trying = df_tfidfvect.stack().reset_index()
trying = trying.rename(columns={0:'tfidf', 'level_0': 'document','level_1': 'term'})
trying = trying.replace("Doc1","poc_comment")
trying = trying.replace("Doc2","white_comment")

trying.sort_values(by=['document','tfidf'], ascending=[True,False]).groupby(['document']).head(10)
```
```
trying[trying['term'].str.contains('angry|rude|emotional|bossy|friendly|aggressive|loud')]
```
```
trying[trying['term'].str.contains('lazy|stupid')]
```
For the first set of words, angry and rude had a higher TF-IDF score for comments under POC Congresswomen’s videos than white Congresswomen. Angry had a score of 0.006481 for POC Congresswomen compared to 0.001148 for white Congresswomen, and rude had 0.009109 and 0. For the terms emotional and loud, the POC video comments had TF-IDF scores of 0, while the white video comments had scores of 0.001613 and 0.004839 respectively. The terms friendly and aggressive did not appear at all in either set of comments. 

Angry is the only term from this list that appears in both set of comments, and the TF-IDF scores suggest that POC Congresswomen are perceived to be more angry. While the scores for emotional, and loud could be indicative of extreme bias, I believe it is more likely due to my sample size. The POC videos contained 235 comments and replies, while the white videos contained 1439 comments. Because the dataset for the white Congresswomen’s videos were much larger and contained more words and these words did not appear in the POC video comments, it can be difficult to draw conclusions about whether emotional and loud are more, less, or equally likely to appear in comments under each type of video. Overall, the data is inconclusive. 

For the other category of words, which are based on broader racial stereotypes instead of gendered ones, there was a notable difference in TF-IDF scores. For POC video comments, the scores for lazy and stupid were 0.012962 and 0.025924 respectively, while they were 0.001149, and 0.008034 for white comments. This data suggests that POC Congresswomen are more likely to be perceived as having characteristics of negative racial stereotypes such as laziness and stupidity, than white Congresswomen are. 

### Discussion & conclusion
There is some limited evidence to suggest my hypothesis is correct, but not strong evidence. I found the results surprising, as I would have expected there to be much larger differences between the TF-IDF scores and as I would have expected the terms in my list to appear a lot more. 

Some limitations of this study are that this study cannot comment on whether women POC policymakers do or do not embody these characteristics more. This project only looks at public perception and comments. Additionally, because I am comparing between women and women of color, I am not providing a baseline for how often these words are used against male members of Congress. In further research, it would be interesting to further explore the gender aspect of my hypothesis by running the methods above with randomly sampled white and POC male congressmen as well. 

Additionally, there are numerous types of selection bias introduced. There may be bias in what personalities of people become members of Congress. As it is a very public facing role, Congressmembers tend to be social and loud, and these criteria could inherently differ based on aspects of identity and what constituents want to elect.

There is also bias introduced in my data collection. As I used the top video result, which tends to be videos with more views, it is likely that comments under videos I am choosing could be more inflammatory than average. YouTube has also introduced a “latest by” feature, where in searching a person’s name, if the person is a content creator it will show the latest video the creator made. A few congresswomen did have YouTube accounts and videos, but to control for what type of video I was selecting, I skipped those and chose the first normally listed video. There were also two videos that were listed as the top video, but that I opted to skip as they had their comments turned off. One video was from Representative Victoria Spartz’s channel and the other was from PBS NewsHour. Spartz’s channel surprised me, as Congressmembers are generally open to commentary and opinions as they have constituents whose opinions they must take into account to be reelected. I was also surprised by PBS NewsHour’s decision to turn off comments, as news media tends to advocate for free speech and discourse. 

Overall, there is some evidence that POC Congresswomen face more biased public perceptions than white Congresswomen do.  In further research, it would be interesting to further develop the study by running the methods above with randomly sampled white and POC male congressmen and on larger sample sizes. 

### References

Chaudhary, Mukesh. “TF-IDF Vectorizer Scikit-Learn.” Medium, Medium, 28 Jan. 2021, medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a. 

“How to Extract YouTube Comments Using YouTube API - Python.” GeeksforGeeks, GeeksforGeeks, 3 Aug. 2023, www.geeksforgeeks.org/how-to-extract-youtube-comments-using-youtube-api-python/. 

OpenAI. (2023). ChatGPT (Dec 17 version) [Large language model]. https://chat.openai.com/share/646d4794-a958-462b-aa13-ac256bdd0857
    Asked "I am using google colab and have written this code, please add a function to store the the for loop outputs in one list"
