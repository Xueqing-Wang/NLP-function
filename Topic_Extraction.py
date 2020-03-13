import pandas as pd
import re
df =pd.read_csv("E:\\OneDrive\\Software\\python_code\\Topic_Extraction\\input\\HillaryEmails.csv",encoding='utf8')
# 清除原邮件数据中的Nan值
df = df[['Id', 'ExtractedBodyText']].dropna()

# 正则表达式 清楚无意义的数据
def clean_email_text(text):
    text = text.replace('\n'," ") #新行，我们是不需要的
    text = re.sub(r"-", " ", text) #把 "-" 的两个单词，分开。（比如：july-edu ==> july edu）
    text = re.sub(r"\d+/\d+/\d+", "", text) #日期，对主体模型没什么意义
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text) #时间，没意义
    text = re.sub(r"[\w]+@[\.\w]+", "", text) #邮件地址，没意义
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text) #网址，没意义
    pure_text = ''
    # 以防还有其他特殊字符（数字）等等，我们直接把他们loop一遍，过滤掉
    for letter in text:
        # 只留下字母和空格
        if letter.isalpha() or letter==' ':
            pure_text += letter
    # 再把那些去除特殊字符后落单的单词，直接排除。
    # 我们就只剩下有意义的单词了。
    text = ' '.join(word for word in pure_text.split() if len(word)>1)
    return text

# docs 邮件正文 #
docs = df['ExtractedBodyText']
docs = docs.apply(lambda s: clean_email_text(s))
print(docs.head(1).values)
doclist = docs.values

from gensim import corpora, models, similarities
import gensim

stoplist = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours',
            'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their',
            'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once',
            'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you',
            'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will',
            'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be',
            'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself',
            'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both',
            'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn',
            'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about',
            'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn',
            'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which']
texts = [[word for word in doc.lower().split() if word not in stoplist] for doc in doclist]
print(texts[0])
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus[3])
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)
print(lda.print_topic(10, topn=5))
print(lda.print_topics(num_topics=20, num_words=5))

# Twitter分类
dd =pd.read_csv("E:\\OneDrive\\Software\\python_code\\Topic_Extraction\\input\\Twitter.csv", encoding='utf8')
dd = dd[['Id','Body']].dropna()
tes = dd['Body']
tes = tes.apply(lambda s: clean_email_text(s))
teslist = tes.values
test = [[word for word in doc.lower().split() if word not in stoplist] for doc in teslist]
print(test[1])
dict = corpora.Dictionary(test)
cc = [dictionary.doc2bow(text) for text in test]
print(cc[0])
topics_test = lda.get_document_topics(cc)
print(topics_test[0], '\n')
print(topics_test[1], '\n')
print(topics_test[2], '\n')
