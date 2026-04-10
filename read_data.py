import os
from fastai.text.all import *
import pandas as pd
from datasets import load_dataset, Dataset

def load_data(path = 'format_data/'):    
    poems = get_text_files(path, folders = ['forms','topics'])
    # print("There are",len(poems),"poems in the dataset")

    poem_files = []
    for p in poems:
        txt = p.open().read()
        poem_files.append(txt)

    print("There are total ", len(poem_files), " poem")
    dataset = Dataset.from_dict({"text": poem_files})
    return poem_files, dataset

def load_topic_data(path = 'format_data/topics/'):
    df = pd.DataFrame(columns = ['poem', 'topic'])
    i = 0

    topics = [name for name in os.listdir(path) if os.path.isdir(path+name)]

    topic_poem = {}

    for t in topics:
        topic_poem[t] = {}
        parray = []
        for file in os.listdir(path+t):
            if file.endswith(".txt"):
                fpath = path+t+'/'+file
                f = open(fpath).read()
                parray.append(f)
        topic_poem[t]['poems'] = parray
        for p in parray:
            df.loc[i] = [p, t]
            i += 1
    return topic_poem, df

def create_topic_prompt_dataset(path="path = 'format_data/topics/'"):
    topic_poem, _ = load_topic_data(path)
    for topic in topic_poem.keys():
        topic_poem[topic]['prompt'] = f"Write a poem about {str(topic)}:"
    return topic_poem
