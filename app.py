import os
import csv
import numpy as np
from sklearn import decomposition
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import extractFeatures
from Bio import SeqIO


import joblib
import pickle

project_root = os.path.dirname(os.path.realpath('__file__'))
template_path = os.path.join(project_root, 'templates')
static_path = os.path.join(project_root, 'static')
app = Flask(__name__, template_folder=template_path, static_folder=static_path)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


class PredForm(Form):
    sequence = TextAreaField("", [validators.DataRequired()])

@app.route("/readme")
def readme():
    return render_template('readme.html', title="ReadMe")
@app.route("/contact")
def contact():
    return render_template('contact.html', title="Contact")
@app.route("/cite")
def cite():
    return render_template('cite.html', title="Citation")
@app.route("/benchmark_data")
def benchmark_data():
    return render_template('benchmark_data.html', title="Benchmark Data")

@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def index():
    form = PredForm(request.form)
    print(form.errors)
    if request.method == 'POST':
        input_seq = form.sequence.data
        savefasta = open(r'myseq.fasta','w+')
        savefasta.write(input_seq)
        savefasta.close()
        storeseq = []#to store fasta sequence
        fasta_sequences = SeqIO.parse(open('myseq.fasta'), 'fasta')
        for fasta in fasta_sequences:
            storeseq.append(str(fasta.seq))


        input_se = storeseq
        #category list to store binding/ non-binding or invalid.
        cate = []
        # to store results of probability of sequence
        prob = []
        pre = []
        p = []
        n = []
        for x in range(len(input_se)):
            if input_se[x].isalpha():
                featur = extractFeatures.get_features([input_se[x]])
                np.random.seed(5)
                inputSize = 153
                outputCol = inputSize + 1

                dataset = np.genfromtxt("./FVs.csv", delimiter=",", dtype=float)
                # print(dataset.shape)
                X = np.array(dataset[:, 0:inputSize], dtype=np.float32)
                X = np.nan_to_num(X)
                print(featur)
                Y = dataset[:, inputSize:outputCol]
                Y = np.nan_to_num(Y)
                std_scale = StandardScaler().fit(X)
                X = std_scale.transform(X)
                X = np.array(X, dtype=np.float32)
                X = np.nan_to_num(X)

                pca = decomposition.PCA(n_components=2)
                pca.fit(X)
                X = pca.transform(X)
                clf = RandomForestClassifier(n_estimators=25, oob_score=True, n_jobs=-1, warm_start=True).fit(X, Y.ravel())
                model = pickle.load(open('model.pkl', 'rb'))
                # model = joblib.load('model.pkl')

                pred = np.round(clf.predict(X))

                featur = np.array(featur, dtype=np.float32)
                featur = np.nan_to_num(featur)
                featur = std_scale.transform(featur)
                featur = np.nan_to_num(featur)
                featur = pca.transform(featur)
                # r = clf.predict_proba(featur)
                pr = model.predict_proba(featur)
                r = model.predict(featur)
                p = pr[:, 1]
                n = pr[:, 0]

                # print(r[0])
                pos = p.tostring()
                neg = n.tostring()
                pos = np.fromstring(pos)
                neg = np.fromstring(neg)
                pos[0] = round(pos[0], 4)
                neg[0] = round(neg[0], 4)

                if r[0] == 1.0:
                    cate.append('BP')
                    # pre = pr[:, 1]
                    prob.append(pos[0])
                if r[0] == 0.0:
                    cate.append('NBP')
                    # pre = pr[:, 0]
                    prob.append(neg[0])
            else:
                cate.append('IN')

        return resultPage(input_se, cate, prob)

    return render_template('home.html', form=form, title="Home")


def resultPage(input_se, cate, prob):
    return render_template('result.html', input_se=input_se, cate=cate, prob=prob, le=len(input_se), title="Results")

if __name__ == "__main__":
    app.run()
