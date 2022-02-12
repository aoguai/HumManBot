"""NLP功能"""
from .SimilarCharactor import similarity

import numpy as np
import re, time


# 检验是否全是中文字符
def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


# 仅保留中文字符
def str_chinese(strs):
    return re.sub('[^\u4e00-\u9fa5]+', '', strs)


# 音形码+余弦相似度
def comprehensive_similar(strs1, strs2):
    # 余弦相似度
    v1, v2 = get_word_vector(strs1, strs2)  # 字符串词向量
    result1 = cos_dist(v1, v2)
    # 音形码相似度

    try:
        result2 = similarity(str_chinese(strs1), str_chinese(strs2))
    except Exception as e:
        print(f"未检测到中文字符\r\n {e}")
        result2 = 0
    if result2 <= 0:
        return result1
    return result1 * 0.5 + result2 * 0.5


def get_word_vector(s1, s2):
    """
    :param s1: 字符串1
    :param s2: 字符串2
    :return: 返回字符串切分后的向量
    """
    # 字符串中文按字分，英文按单词，数字按空格
    regEx = re.compile('[\\W]*')
    res = re.compile(r"([\u4e00-\u9fa5])")
    p1 = regEx.split(s1.lower())
    str1_list = []
    for str in p1:
        if res.split(str) == None:
            str1_list.append(str)
        else:
            ret = res.split(str)
            for ch in ret:
                str1_list.append(ch)
    # print(str1_list)
    p2 = regEx.split(s2.lower())
    str2_list = []
    for str in p2:
        if res.split(str) == None:
            str2_list.append(str)
        else:
            ret = res.split(str)
            for ch in ret:
                str2_list.append(ch)
    # print(str2_list)
    list_word1 = [w for w in str1_list if len(w.strip()) > 0]  # 去掉为空的字符
    list_word2 = [w for w in str2_list if len(w.strip()) > 0]  # 去掉为空的字符
    # 列出所有的词,取并集
    key_word = list(set(list_word1 + list_word2))
    # 给定形状和类型的用0填充的矩阵存储向量
    word_vector1 = np.zeros(len(key_word))
    word_vector2 = np.zeros(len(key_word))
    # 计算词频
    # 依次确定向量的每个位置的值
    for i in range(len(key_word)):
        # 遍历key_word中每个词在句子中的出现次数
        for j in range(len(list_word1)):
            if key_word[i] == list_word1[j]:
                word_vector1[i] += 1
        for k in range(len(list_word2)):
            if key_word[i] == list_word2[k]:
                word_vector2[i] += 1

    # 输出向量
    return word_vector1, word_vector2


def cos_dist(vec1, vec2):
    """
    :param vec1: 向量1
    :param vec2: 向量2
    :return: 返回两个向量的余弦相似度
    """
    dist1 = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return dist1
