# -*- coding: utf-8 -*-
from torch import nn

import torch
import jieba
import numpy as np

raw_text = """越努力就越幸运"""

# 利用jieba进行分词
words = list(jieba.cut(raw_text))
print(words)
# 对标识符去重, 生成由索引:标识符构成的字典
word_to_ix = {i: word for i, word in enumerate(set(words))}
print(word_to_ix)
# 定义嵌入维度, 并用正太分布, 初始化词嵌入
# nn.Embedding 模块的输入是一个标注的下标列表, 输出是对应的词嵌入
embeds = nn.Embedding(4, 3)
print(embeds.weight[0])
# 获取字典的关键字
keys_list = list(word_to_ix.keys())
# 把所有关键字构成的列表转换为张量
tensor_value = torch.LongTensor(keys_list)
# 把张量输入Embedding层, 通过运算得到各标识符的词嵌入
vec = embeds(tensor_value)
print(vec)
