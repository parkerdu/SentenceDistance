import numpy as np
import re

str = ['数码爱好者必备神器 中关村在线客户端 小米note2的大小,不比屏幕大小,拿着和红米note1s是一样大么 浏览数', '234niel是的']


def rm_non_chinese_character(contents):
    """
    删除所有非中文词，包括英文字母、各种符号等。
    注意：该函数在分词操作后调用，因为标点符号对分词结果会有影响。
    """
    sentences = []
    for content in contents:

        content = re.sub(r'[^\u4e00-\u9fa5 ]', '', content)
        content = re.sub(r' +', ' ', content)
        sentences.append(content)
        # content = content.replace(' ', '')
    return sentences
# print(re.sub(r'[^\u4e00-\u9fa5 ]', '', str))
print(rm_non_chinese_character(str))

a = np.array([1, 0, 0])
b = np.array([1, 1, 1])
c = np.dot(a, b)
# print(c)
d = []
d.append('a')
# print(d)
