import pandas as pd
import numpy as np
import re
import snscrape.modules.twitter as sntwitter
from transformers import pipeline
import plotly.express as px 
import joblib
from sklearn.metrics import classification_report,confusion_matrix 


import nltk 
nltk.download("punkt")
nltk.download('stopwords')
from nltk.tokenize import word_tokenize


def get_tweets(username, length=10, option = None):
    # Creating list to append tweet data to
    query = username + " -filter:links filter:replies lang:id"
    if option == "Advanced":
        query = username
    tweets = []
    # Using TwitterSearchScraper to scrape
    # Using TwitterSearchScraper to scrape
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i>=length:
            break
        tweets.append([tweet.content])
    
    # Creating a dataframe from the tweets list above
    tweets_df = pd.DataFrame(tweets, columns=["content"])
    tweets_df['content'] = tweets_df['content'].str.replace('@[^\s]+','')
    tweets_df['content'] = tweets_df['content'].str.replace('#[^\s]+','')
    tweets_df['content'] = tweets_df['content'].str.replace('http\S+','')
    tweets_df['content'] = tweets_df['content'].str.replace('pic.twitter.com\S+','')
    tweets_df['content'] = tweets_df['content'].str.replace('RT','')
    tweets_df['content'] = tweets_df['content'].str.replace('amp','')
    # remove emoticon
    tweets_df['content'] = tweets_df['content'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)

    # remove whitespace leading & trailing
    tweets_df['content'] = tweets_df['content'].str.strip()

    # remove multiple whitespace into single whitespace
    tweets_df['content'] = tweets_df['content'].str.replace('\s+', ' ')

    # remove row with empty content
    tweets_df = tweets_df[tweets_df['content'] != '']
    return tweets_df


def get_sentiment(df,option_model):
    id2label = {0: "negatif", 1: "netral", 2: "positif"}
    if option_model == "IndoBERT (Accurate,Slow)":
        classifier = pipeline("sentiment-analysis",model = "indobert")
        df['sentiment'] = df['content'].apply(lambda x:  id2label[classifier(x)[0]['label']])
    elif (option_model == "Logistic Regression (Less Accurate,Fast)"):
        df_model = joblib.load('assets/df_model.pkl')
        classifier = df_model[df_model.model_name == "Logistic Regression"].model.values[0]
        df['sentiment'] = df['content'].apply(lambda x:  id2label[classifier.predict([x])[0]])
    else :
        df_model = joblib.load('assets/df_model.pkl')
        classifier = df_model[df_model.model_name == option_model].model.values[0]
        df['sentiment'] = df['content'].apply(lambda x:  id2label[classifier.predict([x])[0]])
    # change order sentiment to first column
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    return df

def get_bar_chart(df):
    df= df.groupby(['sentiment']).count().reset_index()
    # plot barchart sentiment
   # plot barchart sentiment
    fig = px.bar(df, x="sentiment", y="content", color="sentiment",text = "content", color_discrete_map={"positif": "#00cc96", "negatif": "#ef553b","netral": "#636efa"})
    # hide legend
    fig.update_layout(showlegend=False)
    # set margin top 
    fig.update_layout(margin=dict(t=0, b=150, l=0, r=0))
    # set title in center
    # set annotation in bar
    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

    # set y axis title
    fig.update_yaxes(title_text='Jumlah Komentar')

    return fig

def plot_model_summary(df_model):
    df_scatter = df_model[df_model.set_data == "test"][["score","time","model_name"]]
    # plot scatter
    fig = px.scatter(df_scatter, x="time", y="score", color="model_name", hover_data=['model_name'])
    # set xlabel to time (s)
    fig.update_xaxes(title_text="time (s)")
    # set ylabel to accuracy 
    fig.update_yaxes(title_text="accuracy")

    # set point size 
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(autosize = False,margin=dict(t=0, l=0, r=0),height = 400)
    return fig

def plot_clfr(df_model,option_model,df):
    df_clfr = pd.DataFrame(classification_report(df["label"],df[f"{option_model}_pred"],output_dict=True)) 
    # heatmap using plotly 
    df_clfr.columns = ["positif","netral","negatif","accuracy","macro_avg","weighted_avg"]
    fig = px.imshow(df_clfr.T.iloc[:,:-1], x=df_clfr.T.iloc[:,:-1].columns, y=df_clfr.T.iloc[:,:-1].index)
    # remove colorbar
    fig.update_layout(coloraxis_showscale=False)
    fig.update_layout(coloraxis_colorscale='gnbu')
    # get annot
    annot = df_clfr.T.iloc[:,:-1].values
    # add annot and set font size 
    fig.update_traces(text=annot, texttemplate='%{text:.2f}',textfont_size=12)
    # set title to classification report 
    fig.update_layout(title_text="📄 Classification Report")
    return fig

def plot_confusion_matrix(df_model,option_model,df):
    # plot confusion matrix 
    cm = confusion_matrix(df['label'],df[f"{option_model}_pred"])
    fig = px.imshow(cm, x=['negatif','netral','positif'], y=['negatif','netral','positif'])
    # remove colorbar
    fig.update_layout(coloraxis_showscale=False)
    fig.update_layout(coloraxis_colorscale='gnbu',title_text = "📊 Confusion Matrix")
    # get annot
    annot = cm
    # add annot
    fig.update_traces(text=annot, texttemplate='%{text:.0f}',textfont_size=15)
    return fig