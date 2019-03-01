# -*- coding: utf-8 -*-
"""
@author: 杜功元
@file: word2vec_demo.py
@time: 2019/02/28
@desc:应用word2vec计算文本相似度
源码参考'https://juejin.im/post/5b237b45f265da59a90c11d6'
"""
import gensim
import re
import jieba
import numpy as np
from scipy.linalg import norm

# 使用bin模型
model_file = 'news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

# # 使用.modle模型
# model_file = 'baike_26g_news_13g_novel_229g.model'
# model = gensim.models.Word2Vec.load(model_file)

def vector_similarity(s1, s2):
    def sentence_vector(s):
        stopwords = load_stopword_list('stopwords.txt')
        words = seg_sentence_rm_stopword(s, stopwords)
        # print(words)
        words = rm_non_chinese_character(words)
        # print(words)
        v = np.zeros(64)
        for word in words:
            v += model[word]
        v /= len(words)
        return v



    def load_stopword_list(filepath):
        """
        载入停用词表。载入从网上找到的包含1893个停用词表。
        此外公司名字里面的"公司","有限责任公司","集团"等信息
        对于行业分类没有作用，也加入停用词表。
        """

        stopwords = ['公司', '有限', '有限责任', '有限公司', '有限责任公司', '集团']
        with open(filepath, 'r', encoding='utf-8') as f:
            stopwords += [line.strip() for line in f.readlines()]
        return stopwords

    def seg_sentence_rm_stopword(sentence, stopwords):
        """
        对句子进行分词，去停用词。

        Arguments:
        sentence -- str, 待处理的句子字符串
        stopwords -- list, 包含停用词的列表

        Returns:
        outstr -- str, 分词并去除停用词后的单词，以空格拼接后的字符串
        """
        sentence = sentence.replace(' ', '')
        sentence_seged = jieba.cut(sentence)

        outstr = []
        for word in sentence_seged:
            if word not in stopwords:
                outstr.append(str(word))
                # outstr += ' '

        return outstr

    def rm_non_chinese_character(contents):
        """
        删除所有非中文词，包括英文字母、各种符号等。
        注意：该函数在分词操作后调用，因为标点符号对分词结果会有影响。
        """
        sentences = []
        for content in contents:
            content = re.sub(r'[^\u4e00-\u9fa5 ]', '', content)
            content = re.sub(r' +', ' ', content)
            if content:
                sentences.append(content)
            # content = content.replace(' ', '')
        return sentences

    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    # print(v1, v2)
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

strings = [
    # '塑胶托板第一品牌-安徽广贤塑业有限公司坐落于安徽省和县西梁山，芜湖长江大桥北边，交通便利。我们是一家集科研，生产，销售于一体的综合型公司，公司专注于开发、生产混凝土砌块机的专用塑胶托板。主营：广贤黑金刚系列免烧砖塑胶托板，是经过多位专业技术人员吸取国内外最新技术开发研制的，具有强度高、韧性强、耐高温，耐腐浊，抗氧化，不怕水，可回收等优点，使用成本远低于老式木制、竹制托板，一经推出就得到国内外客户的一致好评与认可。',
    # '数码爱好者必备神器 中关村在线客户端 小米note2的大小,不比屏幕大小,拿着和红米note1s是一样大么 浏览数',
    # '都是5.5英寸屏幕,只是分辨率不同,红米note是720分辨率,红米note2是1080分辨率。',
    # '你好红米note2分辨率高。具体看如下:红米3主屏:5英寸1280x720像素红米note2主屏:5.5英寸1920x1080像素',
    # '下面具体聊聊屏幕方面的两大重要升级! 百度经验:jingyan.baidu.com 尺寸: 红米是4.7寸,而note竟然蹿升至5.5寸 依旧是IPS屏幕,显示面积却猛增37%,可视角可达178度。这样一来,无论上网还是读书,游戏或者视频,都给你震撼的体验! ...',
    # '红米note和红米note2的屏幕大小区别有图有真相 发表在 求助讨论 来自PC 复制链接 手机看帖 扫一扫!手机看帖更爽 ...',
    # '红米note和魅蓝note2都是一样的5.5寸屏幕,不过整体的色彩饱和度和显示效果魅蓝note2观感更好。',
    #  '红米note2主屏幕5.5英寸,1英寸=2.54厘米,屏幕英寸是指对角线的长度,电脑或电视多少寸也是指屏幕英寸。',
    # '小米红米Note2的屏幕尺寸是5.5英寸。小米红米Note2拥有时尚清新的5种颜色,凡是购买手机的用户,只需要再花费9.9元就可以购买一款彩色后壳。小米红米Note2标配3060mAh大容量可换式电池,续航时间更长',
    # '我爱数码产品'
    # '你在干什么',
    # '你在干啥子',
    # '你在做什么',
    # '你好啊',
    # '我喜欢吃香蕉',
    '前牌照怎么装',
    '如何办理北京车牌',
    '后牌照怎么装',
]

# target = '德州巨润塑料制品有限公司始建于年，坐落于“九达天衢”、“神京门户”之称的德州，成立以来于全国各地公司建立业务合作关系，以先进的管理模式，精湛的技术力量，一流的机械设备，服务于广大客户。我们以质量创新为生命,以人才管理为根本，以市场竞争促成长，本着质量第一，信誉至上的原则，愿在平等互利的条件下，为社会各界朋友提供最优质的服务。我们的不懈努力，赢得了合作企业的信赖与支持。'
target = '车头如何放置车牌'
for string in strings:

    print(string, vector_similarity(string, target))
