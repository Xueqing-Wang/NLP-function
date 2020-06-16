# 网页文本向量化
# 1。 中文语料预处理
from gensim.corpora import WikiCorpus
import jieba

def preprocess ():
    space =''
    i = 0
    l = []
    zhwiki_name = './data/***.xml.bz2'
    f = open('./data/***.txt', 'w')
    wiki = WikiCorpus(zhwiki_name, lemmatize='None', dictionary={})  # xml文件中的训练语料
    for text in wiki.get_texts():
        for temp_sentence in text:
            temp_sentence = Converter('zh_hans').convert(temp_sentence)  # 繁体转为简体
            seg_list = list(jieba.cut(temp_sentence))
            for term in seg_list:
                l.append(term)
        f.write(space.join(l)+'n/')
        i=i+1
        l = []

        if (i%200)==0:
            print("saved"+str(i)+"articles")

    f.close()
# 2。gensim模块训练词向量
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging

logging.basicConfig(format='$(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

def my_function():
    wiki_news = open('./data/***.txt', 'r')
    model = Word2Vec(LineSentence(wiki_news), sg=0, size=192, window=5, min_count=5, workers=9)
    model.save('***')

    if __name__ =='__main__':
        my_function()






