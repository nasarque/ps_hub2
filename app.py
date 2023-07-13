from flask import Flask, request, redirect, url_for, render_template, flash, session, send_from_directory, send_file
from datetime import timedelta, datetime
from sqlalchemy import UniqueConstraint
from flask_wtf import FlaskForm
from flask_migrate import Migrate
from flask_login import login_user, UserMixin, LoginManager, logout_user, login_required
from wtforms import StringField, SubmitField, PasswordField, SelectField
from wtforms.validators import DataRequired, Email
from flask_sqlalchemy import SQLAlchemy
from flask_uploads import UploadSet, configure_uploads, ALL, UploadNotAllowed
import os
import io
import csv
from dotenv import find_dotenv, load_dotenv
import os
#from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from werkzeug.security import generate_password_hash, check_password_hash
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import pyperclip as pc
import time
import pandas as pd
import openai
import tiktoken
#import faiss


# Load environment variables
load_dotenv(find_dotenv())

app = Flask(__name__)

# Use environment variables for secrets
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.permanent_session_lifetime = timedelta(days=5)

db = SQLAlchemy(app)


#-----------------------#

# Tables in my database

class Client(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    client_type = db.Column(db.String(80), nullable=False)

    insight = db.relationship('Insight', backref='client', uselist=False, cascade="all, delete-orphan")
    notes = db.relationship('Note', backref='client', cascade="all, delete-orphan")

class Insight(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    question1 = db.Column(db.String(200), nullable=False)
    question2 = db.Column(db.String(200), nullable=False)
    question3 = db.Column(db.String(200), nullable=False)
    question4 = db.Column(db.String(200), nullable=False)
    question5 = db.Column(db.String(200), nullable=False)
    creator_name = db.Column(db.String(80), nullable=False)

    client_id = db.Column(db.Integer, db.ForeignKey('client.id'), nullable=False)

class Note(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    creator_name = db.Column(db.String(80), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.date)
    content = db.Column(db.Text, nullable=False)

    client_id = db.Column(db.Integer, db.ForeignKey('client.id'), nullable=False)



#------------------#

# Forms


class ClientForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    date = StringField('Date', validators=[DataRequired()], default=datetime.utcnow)
    client_type = SelectField('Client Type', choices=[('sell-side', 'Sell Side'),('buy-side', 'Buy Side'),('capital-markets', 'Capital Markets')], validators=[DataRequired()])

class InsightForm(FlaskForm):
    date = StringField('Date', validators=[DataRequired()])
    question1 = StringField('Question 1', validators=[DataRequired()])
    question2 = StringField('Question 2', validators=[DataRequired()])
    question3 = StringField('Question 3', validators=[DataRequired()])
    question4 = StringField('Question 4', validators=[DataRequired()])
    question5 = StringField('Question 5', validators=[DataRequired()])
    creator_name = StringField('Creator Name', validators=[DataRequired()])
    save = SubmitField('Save')
    complete = SubmitField('Mark as Complete')

#--------------------#

# Routes

@app.route('/insights', methods=['GET', 'POST'])
def insights():
    form = InsightForm()

    if form.validate_on_submit():
        new_insight = Insight(
            client_id=form.client_id.data,
            date=form.date.data,
            question1=form.question1.data,
            question2=form.question2.data,
            question3=form.question3.data,
            question4=form.question4.data,
            question5=form.question5.data,
            creator_name=form.creator_name.data
        )

        db.session.add(new_insight)
        db.session.commit()

        if form.complete.data:
            flash('Insight has been created and marked as complete.')
        else:
            flash('Insight has been created and saved.')
        return redirect(url_for('insights'))

    insights = Insight.query.all()
    return render_template('insights.html', form=form, insights=insights)


@app.route('/client', methods=['GET', 'POST'])
def client():
    form = ClientForm()

    if form.validate_on_submit():
        new_client = Client(
            name=form.name.data,
            date=form.date.data,
            client_type=form.client_type.data
        )

        db.session.add(new_client)
        db.session.commit()

        flash('Client has been created.')
        return redirect(url_for('all_clients'))

    return render_template('client.html', form=form)


@app.route('/all_clients', methods=['GET', 'POST'])
def all_clients():
    return render_template('all_clients.html')



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=False)