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

# �������������Ϊ���ԣ���������κξ�����Ϣ
warnings.filterwarnings("ignore")

# ��������
data_dir = "data"  # ��ȷ�����·����ȷ
stopwords_file = "cn_stopwords.txt"
book_titles_list = [
    "����Х����", "��Ѫ��", "�ɺ��⴫", "���Ǿ�", "¹����",
    "��ʮ������ͼ", "���Ӣ�۴�", "�������", "�齣����¼",
    "�����˲�", "������", "Ц������", "ѩɽ�ɺ�",
    "����������", "ԧ�쵶", "ԽŮ��"
]
K_values = [20, 100, 500, 1000, 3000]  # ���䳤��Kֵ
T_values = [5, 10, 15, 20, 25]  # ��������Tֵ
num_samples = 1000  # ������������
random_seed = 42

# ͣ�ôʶ�ȡ
with open(stopwords_file, encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f])

# �������Ͽ�
corpus = []
labels = []
for book_title in book_titles_list:
    book_path = os.path.join(data_dir, f"{book_title}.txt")
    with open(book_path, encoding='gb18030') as f:
        text = f.read()
    corpus.append(text)
    labels.append(book_title)


# �ִʺ�Ԥ����
def preprocess(text, unit='word'):
    if unit == 'word':
        tokens = jieba.lcut(text)
    else:  # unit == 'char'
        tokens = list(text)
    return ' '.join([token for token in tokens if token not in stopwords and token.strip()])


# �����ȡ����
def extract_paragraphs(text, K):
    tokens = jieba.lcut(text)
    paragraphs = [' '.join(tokens[i:i + K]) for i in range(0, len(tokens), K) if len(tokens[i:i + K]) == K]
    return paragraphs


# �����������ݼ�
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


# LDA + ������ģ�͹ܵ�
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


# ʹ�ý�����֤����ģ��
def evaluate_model(pipeline, X, y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    kf = KFold(n_splits=10, shuffle=True, random_state=random_seed)
    scores = cross_val_score(pipeline, X, y_encoded, cv=kf, scoring='accuracy')
    return scores


# ���ÿ��ģ�͵ķ��౨��
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


# ִ�з�������
def run_classification():
    for unit in ['word', 'char']:
        print(f"���൥λ: {unit}")
        for K in K_values:
            print(f"\n���䳤��K: {K}")
            X, y = generate_samples(K)
            for T in T_values:
                print(f"��������T: {T}")
                pipeline = create_lda_pipeline(T, unit=unit)
                scores = evaluate_model(pipeline, X, y)
                print(f"10�ν�����֤ƽ��׼ȷ��: {np.mean(scores):.4f}")
                print_classification_report(pipeline, X, y)


# ִ��������
if __name__ == "__main__":
    run_classification()
