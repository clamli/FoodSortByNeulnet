from flask import Flask, render_template, request
from wtforms import Form, StringField, validators
from wtforms.validators import DataRequired
import pickle
import sqlite3
import os
import numpy as np

# import HashingVectorizer from local dir
app = Flask(__name__)

######## Preparing the Classifier
cur_dir = os.path.dirname(__file__)
nn = pickle.load(open(os.path.join(cur_dir, 'pkl_objects/classifier.pkl'), 'rb'))
# db = os.path.join(cur_dir, 'reviews.sqlite')

def classify(X):
	y = nn.predict(X)[0]
	return y*(X[0][0]+X[0][1])

# def train(document, y):
# 	X = vect.transform([document])
# 	clf.partial_fit(X, [y])

# def sqlite_entry(path, document, y):
# 	conn = sqlite3.connect(path)
# 	c = conn.cursor()
# 	c.execute("INSERT INTO review_db (review, sentiment, date)"\
# 	" VALUES (?, ?, DATETIME('now'))", (document, y))
# 	conn.commit()
# 	conn.close()


app = Flask(__name__)
class SubmitForm(Form):
	taste = StringField('Taste', validators=[DataRequired()])
	health = StringField('Health', validators=[DataRequired()])

@app.route('/')
def index():
	form = SubmitForm(request.form)
	return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
	form = SubmitForm(request.form)
	if request.method == 'POST' and form.validate():
		print "taste, health"
		taste = int(request.form['taste'])
		health = int(request.form['health'])
		y = classify(np.array([[taste, health]]))
		return render_template('results.html',
								taste=taste,
								health=health,
								prediction=y)
	return render_template('reviewform.html', form=form)

# @app.route('/thanks', methods=['POST'])
# def feedback():
# 	feedback = request.form['feedback_button']
# 	review = request.form['review']
# 	prediction = request.form['prediction']
	
# 	inv_label = {'negative': 0, 'positive': 1}
# 	y = inv_label[prediction]
# 	if feedback == 'Incorrect':
# 		y = int(not(y))
# 	train(review, y)
# 	sqlite_entry(db, review, y)
# 	return render_template('thanks.html')


if __name__ == '__main__':
	app.run("127.0.0.1")