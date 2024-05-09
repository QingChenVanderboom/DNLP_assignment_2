# -*- coding: gb18030 -*-

import os
import random
import warnings

import jieba
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# 将警告输出设置为忽略，即不输出任何警告信息
warnings.filterwarnings("ignore")

# 参数设置
data_dir = "data"  # 请确保这个路径正确
stopwords_file = "cn_stopwords.txt"
book_titles_list = [
    "白马啸西风", "碧血剑", "飞狐外传", "连城诀", "鹿鼎记",
    "三十三剑客图", "射雕英雄传", "神雕侠侣", "书剑恩仇录",
    "天龙八部", "侠客行", "笑傲江湖", "雪山飞狐",
    "倚天屠龙记", "鸳鸯刀", "越女剑"
]
K_values = [20, 100, 500, 1000, 3000]  # 段落长度K值
T_values = [5, 10, 15, 20, 25]  # 主题数量T值
num_samples = 1000  # 抽样段落数量
random_seed = 42

# 停用词读取
with open(stopwords_file, encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f])

# 构建语料库
corpus = []
labels = []
for book_title in book_titles_list:
    book_path = os.path.join(data_dir, f"{book_title}.txt")
    with open(book_path, encoding='gb18030') as f:
        text = f.read()
    corpus.append(text)
    labels.append(book_title)


# 分词和预处理
def preprocess(text, unit='word'):
    if unit == 'word':
        tokens = jieba.lcut(text)
    else:  # unit == 'char'
        tokens = list(text)
    return ' '.join([token for token in tokens if token not in stopwords and token.strip()])


# 随机抽取段落
def extract_paragraphs(text, K):
    tokens = jieba.lcut(text)
    paragraphs = [' '.join(tokens[i:i + K]) for i in range(0, len(tokens), K) if len(tokens[i:i + K]) == K]
    return paragraphs


# 生成样本数据集
def generate_samples(K):
    all_samples = []
    all_labels = []
    for text, label in zip(corpus, labels):
        paragraphs = extract_paragraphs(text, K)
        all_samples.extend(paragraphs)
        all_labels.extend([label] * len(paragraphs))
    combined = list(zip(all_samples, all_labels))
    random.shuffle(combined)
    return zip(*combined[:num_samples])


# LDA + 分类器模型管道
def create_lda_pipeline(n_topics, unit='word'):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=list(stopwords),
                                 token_pattern=r"(?u)\b\w+\b" if unit == 'word' else r".")
    lda = LDA(n_components=n_topics, random_state=random_seed)
    clf = MultinomialNB()
    return Pipeline([
        ('vectorizer', vectorizer),
        ('lda', lda),
        ('clf', clf)
    ])


# 使用交叉验证评估模型
def evaluate_model(pipeline, X, y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    kf = KFold(n_splits=10, shuffle=True, random_state=random_seed)
    scores = cross_val_score(pipeline, X, y_encoded, cv=kf, scoring='accuracy')
    return scores


# 输出每个模型的分类报告
def print_classification_report(pipeline, X, y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    kf = KFold(n_splits=10, shuffle=True, random_state=random_seed)
    y_preds = []
    y_trues = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_preds.extend(y_pred)
        y_trues.extend(y_test)
    report = classification_report(y_trues, y_preds, target_names=encoder.classes_, digits=4)
    print(report)


# 执行分类任务
def run_classification():
    for unit in ['word', 'char']:
        print(f"分类单位: {unit}")
        for K in K_values:
            print(f"\n段落长度K: {K}")
            X, y = generate_samples(K)
            for T in T_values:
                print(f"主题数量T: {T}")
                pipeline = create_lda_pipeline(T, unit=unit)
                scores = evaluate_model(pipeline, X, y)
                print(f"10次交叉验证平均准确率: {np.mean(scores):.4f}")
                print_classification_report(pipeline, X, y)


# 执行主程序
if __name__ == "__main__":
    run_classification()
