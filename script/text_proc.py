import pandas as pd
import numpy as np 
from PIL import Image
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string 
import re #regex library
#umap 
import umap
import hdbscan
import plotly.graph_objects as go
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# import word_tokenize from NLTK
from transformers import AutoTokenizer
from script.plotting import visualize_barchart

def load_stopwords():
    stopwords = pd.read_csv("assets/stopwordbahasa.csv", header=None)
    stopwords = stopwords[0].tolist()
    stopwords = stopwords + list(string.punctuation)
    return stopwords

def tokenisasi(df):
    stopwords = load_stopwords()
    tokenizer = AutoTokenizer.from_pretrained('indobert')
    tokens = df.content.apply(lambda x: tokenizer.tokenize(x))
    tokens = tokens.apply(lambda x: [x for x in x if (not x.startswith('##') and x not in stopwords and len(x) > 4)])
    return tokens

def get_wordcloud(df,kelas_sentiment):
    mask = np.array(Image.open('./assets/twitter.png'))
    cmap_dict = {'positif': 'YlGn', 'negatif': 'OrRd', 'netral': 'GnBu'}
    tokens = tokenisasi(df[df.sentiment == kelas_sentiment])
    tokens = tokens.apply(lambda x: ' '.join(x))
    text = ' '.join(tokens)
    # check if text empty or not 
    try :
        wordcloud = WordCloud(width = 800, height = 800,
                        background_color ='black',
                        min_font_size = 10,
                        colormap = cmap_dict[kelas_sentiment],
                        mask = mask).generate(text)
    except: 
        wordcloud = WordCloud(width = 800, height = 800,
                        background_color ='black',
                        min_font_size = 10,
                        colormap = cmap_dict[kelas_sentiment],
                        mask = mask).generate("None")
    return wordcloud

def plot_text(df,kelas,embedding_model):
    df = df[df.sentiment == kelas]
    data = embedding_model.encode(df.values.tolist())
    umap_model = umap.UMAP(n_neighbors=min(df.shape[0],5),random_state = 42) 
    umap_data = umap_model.fit_transform(data)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=round((df.shape[0])**(0.5)-1),min_samples=3)
    clusterer.fit(umap_data)

    labels = ['cluster ' + str(i) for i in clusterer.labels_]
    # replace cluster -1 with outlier 
    labels = ["outlier" if i == "cluster -1" else i for i in labels ]
    text = df["content"].str.wrap(50).apply(lambda x: x.replace('\n', '<br>'))
    
    fig = px.scatter(x=umap_data[:,0], y=umap_data[:,1],color = clusterer.labels_)
    # remove legend
    fig = px.scatter(x=umap_data[:,0], y=umap_data[:,1],color = labels,text = text)
    #set text color 
    fig.update_traces(textfont_color='rgba(0,0,0,0)',marker_size = 8)
    # set background color
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    # set margin 
    fig.update_layout(margin=dict(l=40, r=5, t=0, b=40))
    # set axis color to grey
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor='rgb(200,200,200)')
    fig.update_yaxes( zeroline=False, linecolor='rgb(200,200,200)')
    # set font sans-serif
    fig.update_layout(font_family="sans-serif")
    # remove legend
    fig.update_layout(showlegend=False)

    # set legend title to cluster
    return df["content"],data,fig

def topic_modelling(df,embed_df):
    data = df.apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    stopwords = load_stopwords()
    # remove empty data 
    topic_model = BERTopic(
        calculate_probabilities=True,
        # cluster model 
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=5,prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words=stopwords),
        language="indonesian",
    )
    topics, probs = topic_model.fit_transform(data,embed_df)
    topic_labels = topic_model.generate_topic_labels(
        topic_prefix = False,
        separator = ", ",
    )
    topic_model.set_topic_labels(topic_labels)
    fig = visualize_barchart(topic_model)
    # set title to Kata Kunci tiap Topic 
    # fig.update_layout(title_text="Topic yang sering muncul")
    return fig,topic_model