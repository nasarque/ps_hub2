from flask import Flask, request, redirect, url_for, render_template, flash, session, send_from_directory, send_file, abort
from datetime import timedelta, datetime
from sqlalchemy import UniqueConstraint
from flask_wtf import FlaskForm
from flask_migrate import Migrate
from flask_login import login_user, UserMixin, LoginManager, logout_user, login_required
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from wtforms import StringField, SubmitField, PasswordField, SelectField, TextAreaField,DateTimeField
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
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


# Load environment variables
load_dotenv(find_dotenv())

app = Flask(__name__)

# Use environment variables for secrets
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.permanent_session_lifetime = timedelta(days=5)

db = SQLAlchemy(app)

admin = Admin(app, url='/admin')

#-----------------------#

# Tables in my database

class Prospect(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    date = db.Column(db.String(80), nullable=False)
    prospect_type = db.Column(db.String(80), nullable=False)

    insight = db.relationship('Insight', backref='prospect', uselist=False, cascade="all, delete-orphan")
    notes = db.relationship('Note', backref='prospect', cascade="all, delete-orphan")

class Insight(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(80), nullable=False)
    question1 = db.Column(db.String(200), nullable=False)
    question2 = db.Column(db.String(200), nullable=False)
    question3 = db.Column(db.String(200), nullable=False)
    question4 = db.Column(db.String(200), nullable=False)
    question5 = db.Column(db.String(200), nullable=False)
    creator_name = db.Column(db.String(80), nullable=False)

    prospect_id = db.Column(db.Integer, db.ForeignKey('prospect.id'), nullable=False)

class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    creator_name = db.Column(db.String(80), nullable=False)
    date = db.Column(db.String(80), nullable=False)
    content = db.Column(db.Text, nullable=False)

    prospect_id = db.Column(db.Integer, db.ForeignKey('prospect.id'), nullable=False)



#------------------#

# Forms


class ProspectForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    date = StringField('Date', validators=[DataRequired()], default=datetime.utcnow().date().isoformat())
    prospect_type = SelectField('prospect Type', choices=[('sell-side', 'Sell Side'),('buy-side', 'Buy Side'),('capital-markets', 'Capital Markets')], validators=[DataRequired()])

class InsightForm(FlaskForm):
    date = StringField('Date', validators=[DataRequired()], default=datetime.utcnow().date().isoformat())
    question1 = StringField('Question 1', validators=[DataRequired()])
    question2 = StringField('Question 2', validators=[DataRequired()])
    question3 = StringField('Question 3', validators=[DataRequired()])
    question4 = StringField('Question 4', validators=[DataRequired()])
    question5 = StringField('Question 5', validators=[DataRequired()])
    creator_name = StringField('Creator Name', validators=[DataRequired()])
    submit = SubmitField('Save')
    complete = SubmitField('Mark as Complete')

class NoteForm(FlaskForm):
    date = StringField('Date', validators=[DataRequired()], default=datetime.utcnow().date().isoformat())
    creator_name = StringField('Creator Name', validators=[DataRequired()])
    content = TextAreaField('Add Note', validators=[DataRequired()])

    submit = SubmitField('Save')
#--------------------#

# Routes

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

# @app.route('/prospect/<int:prospect_id>/add_note', methods=['GET', 'POST'])
# def add_note(prospect_id):
#     prospect = Prospect.query.get(prospect_id)
#     form = NoteForm()
#     if form.validate_on_submit():
#         note = Note(creator_name=form.creator_name.data,
#                     content=form.content.data,
#                     prospect_id=prospect.id,
#                     date=form.date.data
#                     )
#         db.session.add(note)
#         db.session.commit()
#         return redirect(url_for('prospect', prospect_id=prospect.id))
#     return render_template('add_note.html', form=form)


@app.route('/prospect/<int:prospect_id>/add_insight', methods=['GET', 'POST'])
def add_insight(prospect_id):
    prospect = Prospect.query.get(prospect_id)
    form = InsightForm()
    if form.validate_on_submit():
        insight = Insight(
            prospect_id=prospect.id,
            date=form.date.data,
            question1=form.question1.data,
            question2=form.question2.data,
            question3=form.question3.data,
            question4=form.question4.data,
            question5=form.question5.data,
            creator_name=form.creator_name.data
        )

        db.session.add(insight)
        db.session.commit()

        if 'complete' in request.form:
            flash('Insight has been created and marked as complete.')
        elif 'save' in request.form:
            flash('Insight has been created and saved.')
        return redirect(url_for('prospect', prospect_id=prospect.id))
    return render_template('add_insight.html', form=form, prospect=prospect)


@app.route('/prospect/<int:prospect_id>/edit_insight/<int:insight_id>', methods=['GET', 'POST'])
def edit_insight(prospect_id, insight_id):
    prospect = Prospect.query.get(prospect_id)
    insight = Insight.query.get(insight_id)
    if not prospect or not insight:
        abort(404)  # Not found
    form = InsightForm(obj=insight)
    if form.validate_on_submit():
        form.populate_obj(insight)
        db.session.commit()
        flash('Insight has been updated.')
        return redirect(url_for('prospect', prospect_id=prospect.id))
    return render_template('edit_insight.html', form=form, prospect=prospect)


@app.route('/new_prospect', methods=['GET', 'POST'])
def new_prospect():
    form = ProspectForm()

    if form.validate_on_submit():
        new_prospect = Prospect(
            name=form.name.data,
            date=form.date.data,
            prospect_type=form.prospect_type.data
        )

        db.session.add(new_prospect)
        db.session.commit()

        flash('prospect has been created.')
        return redirect(url_for('prospect', prospect_id=new_prospect.id))

    return render_template('new_prospect.html', form=form)


@app.route('/prospect/<int:prospect_id>', methods=['GET'])
def prospect(prospect_id):
    prospect = Prospect.query.get(prospect_id)
    if not prospect:
        abort(404)  # Not found
    return render_template('prospect.html', prospect=prospect)


@app.route('/all_prospects', methods=['GET', 'POST'])
def all_prospects():
    prospects = Prospect.query.all()
    return render_template('all_prospects.html', prospects=prospects)


@app.route('/prospect/<int:prospect_id>/add_note', methods=['GET', 'POST'])
def add_note(prospect_id):
    prospect = Prospect.query.get(prospect_id)
    form = NoteForm()
    if form.validate_on_submit():
        note = Note(creator_name=form.creator_name.data,
                    content=form.content.data,
                    prospect_id=prospect.id,
                    date=form.date.data
                    )
        db.session.add(note)
        db.session.commit()
        return redirect(url_for('prospect', prospect_id=prospect.id))
    return render_template('add_note.html', form=form)


@app.route('/prospect/<int:prospect_id>/edit_note/<int:note_id>', methods=['GET', 'POST'])
def edit_note(prospect_id, note_id):
    prospect = Prospect.query.get(prospect_id)
    note = Note.query.get(note_id)
    if not prospect or not note:
        abort(404)  # Not found
    form = NoteForm(obj=note)
    if form.validate_on_submit():
        form.populate_obj(note)
        db.session.commit()
        flash('Note has been updated.')
        return redirect(url_for('prospect', prospect_id=prospect.id))
    return render_template('edit_note.html', form=form, prospect=prospect)



# Download Notes, Insights




def generate_notes_pdf(prospect_id):
    prospect = Prospect.query.get(prospect_id)

    notes = Note.query.filter_by(prospect_id=prospect_id).all()

    buffer = io.BytesIO()

    # Create the PDF object, using the buffer as its "file."
    pdf = SimpleDocTemplate(buffer, pagesize=letter)

    # Create the list that will hold the flowables
    elements = []

    # Use a built-in style with the name 'Heading1'
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=24,
        leading=30,
        alignment=1,  # Centre align
    )
    elements.append(Paragraph('Notes for Prospect: ' + prospect.name, title_style))

    # Add a spacer
    elements.append(Spacer(1, 0.25 * inch))

    # Add notes
    for note in notes:
        note_style = styles["BodyText"]
        elements.append(Paragraph('<b>Date:</b> ' + note.date, note_style))
        elements.append(Paragraph('<b>Creator Name:</b> ' + note.creator_name, note_style))
        elements.append(Paragraph('<b>Note:</b> ' + note.content, note_style))
        elements.append(Spacer(1, 0.25 * inch))  # Add space after each note

    # Build the PDF
    pdf.build(elements)

    buffer.seek(0)
    return buffer

def generate_insight_pdf(prospect_id):
    prospect = Prospect.query.get(prospect_id)
    insight = Insight.query.filter_by(prospect_id=prospect_id).first()

    buffer = io.BytesIO()

    pdf = SimpleDocTemplate(buffer, rightMargin=inch/2, leftMargin=inch/2, topMargin=inch/2, bottomMargin=inch/2)
    elements = []

    title_style = getSampleStyleSheet()['Title']
    insight_style = getSampleStyleSheet()['BodyText']

    elements.append(Paragraph('Insight for Prospect: ' + prospect.name, title_style))

    elements.append(Paragraph(f"Creator: {insight.creator_name}", insight_style))
    elements.append(Paragraph(f"Date: {insight.date}", insight_style))
    elements.append(Paragraph(f"Question 1: {insight.question1}", insight_style))
    elements.append(Paragraph(f"Question 2: {insight.question2}", insight_style))
    elements.append(Paragraph(f"Question 3: {insight.question3}", insight_style))
    elements.append(Paragraph(f"Question 4: {insight.question4}", insight_style))
    elements.append(Paragraph(f"Question 5: {insight.question5}", insight_style))

    pdf.build(elements)

    buffer.seek(0)
    return buffer

def generate_summary_pdf(prospect_id):
    prospect = Prospect.query.get(prospect_id)
    notes = Note.query.filter_by(prospect_id=prospect_id).all()
    insight = Insight.query.filter_by(prospect_id=prospect_id).first()

    buffer = io.BytesIO()

    pdf = SimpleDocTemplate(buffer, rightMargin=inch/2, leftMargin=inch/2, topMargin=inch/2, bottomMargin=inch/2)
    elements = []

    title_style = getSampleStyleSheet()['Title']
    note_style = getSampleStyleSheet()['BodyText']
    insight_style = getSampleStyleSheet()['BodyText']

    # Add the notes
    elements.append(Paragraph('Notes for Prospect: ' + prospect.name, title_style))
    for note in notes:
        elements.append(Paragraph(f"Creator: {note.creator_name}", note_style))
        elements.append(Paragraph(f"Date: {note.date}", note_style))
        elements.append(Paragraph(f"Content: {note.content}", note_style))
        elements.append(Spacer(1, 0.2*inch))

    elements.append(PageBreak())

    # Add the insight
    elements.append(Paragraph('Insight for Prospect: ' + prospect.name, title_style))
    elements.append(Paragraph(f"Creator: {insight.creator_name}", insight_style))
    elements.append(Paragraph(f"Date: {insight.date}", insight_style))
    elements.append(Paragraph(f"Question 1: {insight.question1}", insight_style))
    elements.append(Paragraph(f"Question 2: {insight.question2}", insight_style))
    elements.append(Paragraph(f"Question 3: {insight.question3}", insight_style))
    elements.append(Paragraph(f"Question 4: {insight.question4}", insight_style))
    elements.append(Paragraph(f"Question 5: {insight.question5}", insight_style))

    #Build the pdf
    pdf.build(elements)

    buffer.seek(0)
    return buffer

@app.route('/download_notes/<prospect_id>', methods=['GET'])
def download_notes(prospect_id):
    pdf = generate_notes_pdf(prospect_id)
    return send_file(pdf, download_name='notes.pdf', as_attachment=True)


@app.route('/download_insight/<prospect_id>', methods=['GET'])
def download_insight(prospect_id):
    pdf = generate_insight_pdf(prospect_id)
    return send_file(pdf, download_name='insight.pdf', as_attachment=True)


@app.route('/download_summary/<prospect_id>', methods=['GET'])
def download_summary(prospect_id):
    pdf = generate_summary_pdf(prospect_id)
    return send_file(pdf, download_name='summary.pdf', as_attachment=True)



admin.add_view(ModelView(Prospect, db.session))
admin.add_view(ModelView(Note, db.session))
admin.add_view(ModelView(Insight, db.session))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=False)