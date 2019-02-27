from flask import Flask, render_template, flash, request, redirect, url_for
from wtforms import Form, TextField, TextAreaField, validators, StringField, DecimalField, SubmitField

import requests
import urllib
from bs4 import BeautifulSoup
from readability.readability import Document # https://github.com/buriy/python-readability. Tried Goose, Newspaper (python libraries on Github). Bad results.
from http.cookiejar import CookieJar #
import json
import numpy as np

import nltk
from nltk import *
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from newspaper import Article

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'index'

class UrlForm(Form):
    url = TextField('Enter article URL:', validators=[validators.DataRequired()])
    operatingExpense = DecimalField('Enter Operating Expense:', validators=[validators.DataRequired()])
    grossProfit = DecimalField('Enter Gross Profit:', validators=[validators.DataRequired()])
    profitMargin = DecimalField('Enter Profit Margin:', validators=[validators.DataRequired()])
    submit = SubmitField('Calculate')

@app.route("/", methods=['GET', 'POST'])
def hello():
    form = UrlForm(request.form)
    if request.method == 'POST':
        url=request.form['url']
        operatingExpense=request.form['operatingExpense']
        grossProfit=request.form['grossProfit']
        profitMargin=request.form['profitMargin']
        print(url, grossProfit, profitMargin)
        if form.validate():
            print(url, grossProfit, profitMargin)
            return redirect(url_for('view', url=url, operatingExpense=operatingExpense,
                                    grossProfit=grossProfit, profitMargin=profitMargin))
    return render_template('index.html', form=form)

def sentiment_polarity(s):
    paragraphs = re.split('\n' , s)
    paragraphs = [p for p in paragraphs if p != '']
    sid = SentimentIntensityAnalyzer()
    neu = []
    compound = []
    for sentence in paragraphs:
        ss = sid.polarity_scores(sentence)
        neu.append(ss['neu'])
    return np.var(neu)

@app.route('/view/<path:url>')
def view(url):
    title = ""
    authors = ""
    publish_date = ""
    text = ""
    top_image = ""


    # opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor)
    # opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/535.7 (KHTML, like Gecko) Chrome/16.0.912.77 Safari/535.7')]
    # html =  opener.open(url).read().decode('utf-8')
    # readable_article = Document(html).summary()
    # soup = BeautifulSoup(readable_article, "lxml")
    # text = soup.get_text()
    article = Article(url)
    article.download()
    article.parse()

    title = article.title
    authors = article.authors
    publish_date = article.publish_date
    text = article.text
    top_image = article.top_image
    sentiment = sentiment_polarity(text)

    return render_template('view.html', title = title, authors = authors, publish_date=publish_date, text=text, top_image=top_image, sentiment=sentiment)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
