"""
File: main.py
Author: Chuncheng Zhang
Date: 2023-07-17
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2023-07-17 ------------------------
# Requirements and constants
# import torch
import numpy as np
import pandas as pd
# import plotly.io as pio
import plotly.express as px

# import seaborn as sns
# import matplotlib as mpl
import matplotlib.pyplot as plt

from rich import print, inspect
from sklearn.cluster import KMeans
from sklearn.manifold import Isomap
# from sklearn.decomposition import PCA
from collections import defaultdict
from transformers import AutoTokenizer, BertModel

from IPython.display import display

# %%
plotly_template = 'seaborn'
plotly_draw_flag = True
plotly_textfont_size = 15

# %%
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")


# %% ---- 2023-07-17 ------------------------
# Function and class
def compute(text):
    """Compute text for its state

    Args:
        text (str): Input text.

    Returns:
        tokens, inputs, outputs
    """
    print('--------------------------------')
    print('Compute {}'.format(text))
    tokens = defaultdict(list)
    tokens_in_order = []

    # Summary the tokens
    if isinstance(text, list):
        status_vectors_lst = []
        for sentence_idx, text_row in enumerate(text):
            inputs = tokenizer(text_row, return_tensors="pt")

            for token in inputs['input_ids'][0]:
                token_id = int(token)
                tokens_in_order.append((sentence_idx, token_id))
                tokens[token_id].append(tokenizer.decode(token))

            # Compute the status
            outputs = model(**inputs)
            status_vectors_lst.append(
                outputs.last_hidden_state.detach().cpu().numpy().squeeze())

        status_vectors = np.concatenate(status_vectors_lst, axis=0)

    if isinstance(text, str):
        inputs = tokenizer(text, return_tensors="pt")

        for token in inputs['input_ids'][0]:
            token_id = int(token)
            tokens_in_order.append((0, token_id))
            tokens[token_id].append(tokenizer.decode(token))

        # Compute the status
        outputs = model(**inputs)
        status_vectors = outputs.last_hidden_state.detach().cpu().numpy().squeeze()

    print(tokens)

    # Embedding the status
    # 1. Embed into Isomap components
    # 2. Embed into kmeans id
    embedding = Isomap(n_components=2)
    embedding_components = embedding.fit_transform(status_vectors)

    kmeans = KMeans(n_clusters=5)
    cluster_id = kmeans.fit_predict(status_vectors)

    # Summary the table
    lst = []
    token_idx = 0
    for sentence_idx, token_id in tokens_in_order:
        lst.append((sentence_idx, token_idx, token_id))
        if token_id == 102:
            token_idx = 0
        else:
            token_idx += 1

    table = pd.DataFrame(lst, columns=['sentenceIdx', 'tokenIdx', 'tokenId'])
    for k in ['sentenceIdx', 'tokenIdx']:
        table[k] = table[k].map(str)
    table['token'] = table['tokenId'].map(tokenizer.decode)
    table['feature-1'] = embedding_components[:, 0]
    table['feature-2'] = embedding_components[:, 1]
    table['clusterId'] = [str(e) for e in cluster_id]
    table['status'] = [e for e in status_vectors]
    display(table)

    # Plot the corrcoef
    figs = []
    corr = np.corrcoef(status_vectors)
    fig = px.imshow(corr, template=plotly_template,
                    x=table['token'], y=table['token'],
                    width=800, height=800, title='Corrcoef')
    figs.append(fig)

    # Plot the Isomap space
    # fig = px.scatter(table, x='feature-1', y='feature-2',
    #                  text='token', color='clusterId',
    #                  width=800, height=800,
    #                  template=plotly_template,
    #                  title='Tokens in Isomap')
    # fig.update_traces(textposition='top center')
    # fig.update_scenes(aspectratio=dict(x=1, y=1))
    # figs.append(fig)

    if plotly_draw_flag:
        for fig in figs:
            fig.show()

    return tokens, inputs, outputs, table


def draw_table(table):
    """Draw the table with three views

    Args:
        table (DataFrame): The table to draw.

    Returns:
        list: The list of the plots.
    """
    figs = []

    # Plot the Isomap space
    fig = px.scatter(table, x='feature-1', y='feature-2',
                     text='token', color='sentenceIdx',
                     width=800, height=800, template=plotly_template, title='Tokens in Isomap by sentenceIdx')
    # fig.update_traces(textposition='top center')
    # fig.update_scenes(aspectratio=dict(x=1, y=1))
    figs.append(fig)

    fig = px.line(table, x='feature-1', y='feature-2',
                  text='token', color='clusterId',
                  width=800, height=800,
                  template=plotly_template,
                  title='Tokens in Isomap by clusterId')
    # fig.update_traces(textposition='top center', textfont=dict(size=15))
    # fig.update_scenes(aspectratio=dict(x=1, y=1))
    figs.append(fig)

    fig = px.line(table, x='feature-1', y='feature-2',
                  text='token', color='token',
                  width=800, height=800,
                  template=plotly_template,
                  title='Tokens in Isomap by token')
    # fig.update_traces(textposition='top center', textfont=dict(size=15))
    # fig.update_scenes(aspectratio=dict(x=1, y=1))
    figs.append(fig)

    # Plot the multiple col graphs
    fig = px.line(table, x='feature-1', y='feature-2',
                  text='token',
                  facet_col='sentenceIdx', template=plotly_template,
                  width=800, height=400, title='Tokens in Isomap')
    # fig.update_traces(textposition='top center')
    # fig.update_scenes(aspectratio=dict(x=1, y=1))
    figs.append(fig)

    fig = px.scatter(table, x='feature-1', y='feature-2',
                     text='token', color='clusterId',
                     facet_col='sentenceIdx',
                     width=800, height=400, template=plotly_template, title='Tokens in Isomap')
    # fig.update_traces(textposition='top center')
    # fig.update_scenes(aspectratio=dict(x=1, y=1))
    figs.append(fig)

    if plotly_draw_flag:
        for fig in figs:
            fig.update_traces(textposition='top center',
                              textfont=dict(size=plotly_textfont_size))
            fig.update_scenes(aspectratio=dict(x=1, y=1))
            fig.show()

    status = np.array([list(e) for e in table['status'].values])
    fig = px.imshow(status, template=plotly_template)
    fig.show()

    return figs


# %% ---- 2023-07-17 ------------------------
# Play ground
text1 = '''
根据《关于促进服务业领域困难行业纾困发展有关增值税政策的公告》有关规定： 2022年1月1日至2022年12月31日，网约车服务免征增值税。滴滴对于2022年1月1日至2022年12月31日为您提供的网约车服务将为您开具免税字样的增值税普通发票。
'''
tokens1, inputs1, outputs1, table1 = compute(text1)

text2 = '''
根据《关于促进服务业领域困难行业纾困发展有关增值税政策的公告》有关规定
'''
tokens2, inputs2, outputs2, table2 = compute(text2)


text3 = '''
Many words map to one token, but some don't: indivisible.

Unicode characters like emojis may be split into many tokens containing the underlying bytes: 🤚🏾

Sequences of characters commonly found next to each other may be grouped together: 1234567890
'''
tokens3, inputs3, outputs3, table3 = compute(text3)

figs = draw_table(table3)

# %% ---- 2023-07-17 ------------------------
# Pending
text = [text1, text2, text3]
tokens, inputs, outputs, table = compute(text)

figs = draw_table(table)

# draw_table_overall(table)

# %%

text = [
    '''
    汽车的生命是汽油
    ''',
    '''
    汽车的汽油是生命
    ''',
    '''
    生命的汽车是汽油
    ''',
    '''
    汽油是石油的副产品
    ''',
    '''
    石油曾经是石头
    '''
]

tokens, inputs, outputs, table = compute(text)
figs = draw_table(table)

# draw_table_overall(table)

# %%


# %%
# %%
