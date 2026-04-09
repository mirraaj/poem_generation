import os
from fastai.text.all import *
import pandas as pd
from datasets import load_dataset, Dataset
from pathlib import Path

def format_data(path = 'data/'):    
    poems = get_text_files(path, folders = ['forms','topics'])
    # print("There are",len(poems),"poems in the dataset")

    poem_files = []
    for p in poems:
        txt = p.open().read()
        formatted = f"""<POEM>
{txt}
</POEM>"""
        print(p)
        newpath = "format_"+str(p)
        file = Path(newpath)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(formatted)
        # print(newpath)
        # with open(newpath, "w") as f:
        #     f.write(formatted)

    
if __name__=="__main__":
    format_data()