def get_content(path):
    with open(path, 'r', encoding='gbk', errors='ignore')as f:
        content = ''
        for l in f:
            l = l.strip()
            content += l
        return content


def get_TF(words, topK=10):
    tf_dict={}
    for w in words:
        dict[w] = tf_dict.get(w, 0)+1
    return sorted(tf_dict.items(), key=lambda x: x[1], reverse=True)[:topK]

def main ():
    import glob
    import random
    import jieba
    files = glob.glob('.///')
    corpus = [get_content(x) for x in files]  # x是文件名，glob返回值是文件名的列表

    index = random.randint(0, len(corpus))
    split_sample = list(jieba.cut(corpus[index]))
    print("样本之一："+corpus[index])
    print("样本分词效果："+'/'.join(split_sample))
    print('Top 10高频词'+str(get_TF(split_sample)))
