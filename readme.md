# LLM 中 Token 的通俗解释

本文使用 python 的 transformers 包提供的预训练模型进行 token 解析，并尝试通过解析结果来回答 token 是什么的问题。通过几个例子看到，在不同的语境下，相同的 token 经过语言模型计算之后，可以得到不同的特征向量。这说明 LLM 在 token 的特征向量这一层级已经开始对语义信息进行处理，处理的基础是 token 对应的特征向量。

本文开源代码可见我的 Github 仓库

[https://github.com/listenzcc/learn-tokenizer](https://github.com/listenzcc/learn-tokenizer)

---
- [LLM 中 Token 的通俗解释](#llm-中-token-的通俗解释)
  - [LLM 的 token](#llm-的-token)
  - [Token 与向量](#token-与向量)
  - [Token 向量的通俗解释](#token-向量的通俗解释)
  - [中英文 token 特征的对比](#中英文-token-特征的对比)
  - [附录：Token 计算代码](#附录token-计算代码)


## LLM 的 token

Token 在大语言模型（LLM）中有特殊的意义，是 LLM 进行处理的最小单元。ChatGPT 模型对 token 的解释是

> The GPT family of models process text using **tokens**, which are common sequences of characters found in text. The models understand the statistical relationships between these tokens, and excel at producing the next token in a sequence of tokens.
> 

将自然语言转换为 token 的过程称为 tokenization

> Tokenization is the process of splitting the input and output texts into smaller units that can be processed by the LLM AI models. Tokens can be words, characters, subwords, or symbols, depending on the type and the size of the model.
> 

BERT 模型对 token 的使用方法是

> BERT uses tokens as the basic input units to process text. These tokens are generated through the process of tokenization, which involves breaking down the input text into smaller subword units or characters.
> 

## Token 与向量

本文使用 python 的 transformers 包提供的预训练模型进行 token 解析，并尝试通过解析结果来回答 token 是什么的问题，模型部署的核心代码如附录所示。以一段英文为例

> Many words map to one token, but some don't: indivisible.
> 
> 
> Unicode characters like emojis may be split into many tokens containing the underlying bytes: 🤚🏾
> 
> Sequences of characters commonly found next to each other may be grouped together: 1234567890
> 

它的 token 既不是以单词为单位，也不是以单词的组合为单位，而是以单词的词根为单位，被切分为模型能够认识的 token，如下图所示。图中每一个点都代表一个 token，它们既有完整的单词，又有井号开头的词根，还有[CLS]、[UNK]、[SEP]等代表的特殊字符。

![Untitled](Token%20%E7%9A%84%E9%80%9A%E4%BF%97%E8%A7%A3%E9%87%8A%204ab476110d9f443ba27f6f982560ebe9/Untitled.png)

我们从模型得到的特征并不是图中横纵坐标表示的二维向量（feature-1 和 feature-2），而是 1024 维的向量，如下图所示。我是通过降维算法 Isomap 将 1024 维的向量降维到 2 维空间中对它们进行展示，形成上图。

![Untitled](Token%20%E7%9A%84%E9%80%9A%E4%BF%97%E8%A7%A3%E9%87%8A%204ab476110d9f443ba27f6f982560ebe9/Untitled%201.png)

## Token 向量的通俗解释

为了进一步解释说明 token 是什么，我使用另外一个例子，那就是用上述处理方法对以下语句进行处理

- 汽车的生命是汽油
- 汽车的汽油是生命
- 生命的汽车是汽油
- 汽油是石油的副产品
- 石油曾经是石头

对这 5 句话进行 token 分析后得到下图，其中左图是按句子顺序着色的 token 位置，右图是这些 token 的 1024 维向量之间的互相关矩阵。

![Untitled](Token%20%E7%9A%84%E9%80%9A%E4%BF%97%E8%A7%A3%E9%87%8A%204ab476110d9f443ba27f6f982560ebe9/Untitled%202.png)

![Untitled](Token%20%E7%9A%84%E9%80%9A%E4%BF%97%E8%A7%A3%E9%87%8A%204ab476110d9f443ba27f6f982560ebe9/Untitled%203.png)

为了清楚起见，我将上图中的左图放大如下，并且按 token 对点进行着色，还进一步将这些点连接起来，得到下图。从下图中可以看到，虽然每个分句的 token 有重复的成分，但经过语言模型计算之后，相同的 token 对应不同的特征向量。这说明在不同的语境下，相同的 token 经过语言模型计算之后，可以得到不同的特征向量。

![Untitled](Token%20%E7%9A%84%E9%80%9A%E4%BF%97%E8%A7%A3%E9%87%8A%204ab476110d9f443ba27f6f982560ebe9/Untitled%204.png)

接下来，我将不同的句子分开绘图，并且将这些 token 按出现的顺序连接起来，得到一些曲线。可以看到，每条曲线都是以 [CLS] 为起点，以 [SEP] 为终点的可变长度拆线，长度为该句中的 token 数量。

![Untitled](Token%20%E7%9A%84%E9%80%9A%E4%BF%97%E8%A7%A3%E9%87%8A%204ab476110d9f443ba27f6f982560ebe9/Untitled%205.png)

最后，我对这些向量进行简单的 KMeans 聚类分析，用类别对点进行着色，可以看到它们在 1024 维的特征空间呈现语义相近的匹配关系。

![Untitled](Token%20%E7%9A%84%E9%80%9A%E4%BF%97%E8%A7%A3%E9%87%8A%204ab476110d9f443ba27f6f982560ebe9/Untitled%206.png)

![Untitled](Token%20%E7%9A%84%E9%80%9A%E4%BF%97%E8%A7%A3%E9%87%8A%204ab476110d9f443ba27f6f982560ebe9/Untitled%207.png)

## 中英文 token 特征的对比

下方的互相关矩阵图代表中英文夹杂的两段话的 token 向量之间的相似关系，可以看到中文与英文彼此的相似程度小于语言内的相似程度。

右图是映射到 2 维空间的聚类图，图中可见英文 token 集中在右下角、数字 token 集中在左下角，其余为中文 token。这说明 LLM 在 token 的特征向量这一层级已经开始对语义信息进行处理，处理的基础是 token 对应的特征向量。

![Untitled](Token%20%E7%9A%84%E9%80%9A%E4%BF%97%E8%A7%A3%E9%87%8A%204ab476110d9f443ba27f6f982560ebe9/Untitled%208.png)

![Untitled](Token%20%E7%9A%84%E9%80%9A%E4%BF%97%E8%A7%A3%E9%87%8A%204ab476110d9f443ba27f6f982560ebe9/Untitled%209.png)

## 附录：Token 计算代码

```python
'''
Example for tokenization of input text
'''

# Import modules
from transformers import AutoTokenizer, BertModel
from IPython.display import 

# Load pre-trained model ~ hundreds MB size.
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")

# Setup input text
text_input = '<-- Your input text goes here -->'

# Generate model inputs as kwargs,
# it tokenizes the text_input.
inputs = tokenizer(text_input, return_tensors="pt")

# Forward the inputs
outputs = model(**inputs)

# The last hidden state refers the feature of the tokens
features = outputs.last_hidden_state
```