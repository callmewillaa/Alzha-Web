from flask import Flask, render_template, request, redirect, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import pickle
import joblib
import random
import json
from keras.models import load_model
import cv2
import os
import numpy as np
import tensorflow as tf
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from datetime import datetime
from flask_mail import Mail, Message
import re
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cuy.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'

class Klasifikasi(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nama = db.Column(db.String(100), nullable=False)
    tanggal = db.Column(db.String(100), default=datetime.utcnow, nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    umur = db.Column(db.Integer, nullable=False)
    jenis_kelamin = db.Column(db.String(10), nullable=False)
    image_path = db.Column(db.String(255), nullable=True)
    klasifikasi_class = db.Column(db.String(50), nullable=True)
    probabiliti = db.Column(db.Float, nullable=True)

class RiwayatKlasifikasi(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nama = db.Column(db.String(100), nullable=False)
    tanggal = db.Column(db.String(100), default=datetime.utcnow, nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    umur = db.Column(db.Integer, nullable=False)
    jenis_kelamin = db.Column(db.String(10), nullable=False)
    image_path = db.Column(db.String(255), nullable=True)
    klasifikasi_class = db.Column(db.String(50), nullable=True)
    probabiliti = db.Column(db.Float, nullable=True)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nama = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    
    def check_password(self,password):
        return password == self.password
    
class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    comment = db.Column(db.String(255), nullable=False)
    sentiment = db.Column(db.String(10), nullable=False)

with app.app_context():
    db.create_all()


app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your email'
app.config['MAIL_DEFAULT_SENDER'] = 'your email'
app.config['MAIL_PASSWORD'] = 'your password'

mail = Mail(app)

@app.route('/kirim-email-dashboard', methods=['POST'])
def kirim_email_dashboard():
    email = request.form['email']
    riwayat_id = request.form['riwayat_id']  # Ubah nama sesuai dengan yang ada di HTML

    # Ambil data riwayat klasifikasi dari database berdasarkan ID
    riwayat_klasifikasi = RiwayatKlasifikasi.query.get(riwayat_id)

    # Kirim email dengan hasil klasifikasi
    subject = "Hasil Klasifikasi Alzheimer"
    body = f"Hasil Klasifikasi Alzheimer untuk pasien Dengan :\n" \
       f"Nama : {riwayat_klasifikasi.nama}\n" \
       f"Umur : {riwayat_klasifikasi.umur}\n" \
       f"Dengan Jenis Kelamin : {riwayat_klasifikasi.jenis_kelamin}\n" \
       f"Pada tanggal : {riwayat_klasifikasi.tanggal}\n" \
       f"Jenis Alzheimer : {riwayat_klasifikasi.klasifikasi_class} dengan probabilitas {riwayat_klasifikasi.probabiliti}%" 
       

    msg = Message(subject, recipients=[email], body=body)
    mail.send(msg)

    return redirect('/dashboard')
    
# Memuat model chatbot dan informasi yang diperlukan
lemmatizer = WordNetLemmatizer()
chatbot = load_model("model-chatbot.h5")
intents = json.loads(open('dataset.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

def preprocess_text(text):
    # Case Folding
    text = text.lower()

    # Filtering
    text = re.sub(r"""(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))"""," ", text)
    text = re.sub(r"\(cont\)", " ", text)
    text = re.sub('[!"”#$%&’()*+,-./:;<=>?@[\\]^_`{|}~]', " ", text)
    text = re.sub(r"#([^\s]+)", "", text)
    text = re.sub(r"\d+", " ", text)

    return text

def remove_stopwords(tokens):
    factory = StopWordRemoverFactory()
    stopword_list = factory.get_stop_words()
    return [word for word in tokens if word not in stopword_list and len(word) > 3]

def replace_slang(word, dataslang):
    if word in list(dataslang[0]):
        indexslang = list(dataslang[0]).index(word)
        return dataslang[1][indexslang]
    else:
        return word

def stemmer(line, ind_stemmer):
    temp = list()
    for word in line:
        word = ind_stemmer.stem(word)
        if len(word) > 2:
            temp.append(word)
    return temp

def predict_sentiment(text, vectorizer, load_model):
    test_data = [str(text)]
    test_vector = vectorizer.transform(test_data).toarray()
    pred = load_model.predict(test_vector)

    # Konversi nilai biner menjadi kategori
    if pred[0] == 1:
        sentiment_category = 'negatif'
    elif pred[0] == 3:
        sentiment_category = 'netral'
    else :
        sentiment_category = 'positif'

    return sentiment_category
    

def unique_labels(labels):
    return set(labels)

def create_sentiment_chart(positive_count, negative_count, netral_count):
    labels = ['Positif', 'Negatif', 'Netral']
    counts = [positive_count, negative_count, netral_count]

    fig, ax = plt.subplots()
    ax.bar(labels, counts, color=['green', 'red', 'yellow'])

    # Tambahkan label dan judul
    ax.set_ylabel('Jumlah')
    ax.set_title('Perbandingan Jumlah Sentimen')

    # Simpan chart sebagai gambar
    chart_path = 'static/sentimen/sentiment_chart.png'
    plt.savefig(chart_path)

    return chart_path

def clean_up_sentence(sentence):
    # tokenize memecah kalimat menjadi perkata
    sentence_words = nltk.word_tokenize(sentence)
    # mengubah kalimat menjadi bentuk dasar
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, chatbot):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = chatbot.predict(np.array([p]))[0]
    THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, chatbot)
    res = getResponse(ints, intents)
    return res

model = tf.keras.models.load_model('mobilenet-alzheimer.h5')  # load 'alzheimer_model.h5' 
class_labels = ['Alzheimer Sangat Ringan', 'Tidak Terkena Alzheimer', 'Alzheimer Sedang', 'Alzheimer Ringan']

def klasifikasi_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = np.array(img).reshape(-1, 224, 224, 3)
    prediction = model.predict(img)
    klasifikasi_class = class_labels[np.argmax(prediction)]
    probabiliti = max(prediction[0]) * 100
    return klasifikasi_class, probabiliti

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        
        return redirect("/")
    else:
        
        return render_template("index.html")


@app.route("/pasien")
def pasien():
    return render_template("pasien.html")

@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == 'POST':
        nama = request.form['nama']
        email = request.form['email']
        password = request.form['password']
        new_user = User(nama=nama,email=email,password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
    return render_template('register.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['email'] = user.email 
            return redirect('/dashboard')
        else:
            return render_template('login.html',error='Invalid user')
    return render_template('login.html')

@app.route("/get_users", methods=['GET'])
def get_users():
    users = User.query.all()
    user_list = [{'nama': user.nama, 'email': user.email, 'password' : user.password} for user in users]
    return {'users': user_list}

@app.route('/logout')
def logout():
    session.pop('email')
    return redirect('/login')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':

        nm = request.form.get('nama', '')
        tgl = request.form.get('tanggal', '')
        email = request.form.get('email', '')
        umr = request.form.get('umur', '')
        jk = request.form.get('jk', '')
        file = request.files.get('file')
        file_path = None

        # Cek apakah alamat email sudah ada dalam database Klasifikasi
        existing_klasifikasi = Klasifikasi.query.filter_by(email=email).first()

        if existing_klasifikasi:
            # Jika email sudah ada, perbarui entri yang sudah ada
            existing_klasifikasi.tanggal = tgl
            existing_klasifikasi.umur = umr
            existing_klasifikasi.jenis_kelamin = jk
            db.session.commit()
        else:
            # Jika email belum ada, buat entri baru dalam database Klasifikasi
            new_klasifikasi = Klasifikasi(nama=nm, tanggal=tgl, email=email, umur=umr, jenis_kelamin=jk)
            db.session.add(new_klasifikasi)
            db.session.commit()

        if file:
            filename = file.filename
            file_path = os.path.join('static/uploads', filename)
            file.save(file_path)
            klasifikasi_class, probabiliti = klasifikasi_image(file_path)

            # Update data di dalam database Klasifikasi
            existing_klasifikasi.image_path = file_path
            existing_klasifikasi.klasifikasi_class = klasifikasi_class
            existing_klasifikasi.probabiliti = probabiliti
            db.session.commit()

            # Periksa apakah email sudah ada dalam database RiwayatKlasifikasi
            existing_riwayat = RiwayatKlasifikasi.query.filter_by(email=email).first()

            if existing_riwayat:
                # Jika email sudah ada, perbarui entri yang sudah ada
                existing_riwayat.tanggal = tgl
                existing_riwayat.umur = umr
                existing_riwayat.jenis_kelamin = jk
                existing_riwayat.image_path = file_path
                existing_riwayat.klasifikasi_class = klasifikasi_class
                existing_riwayat.probabiliti = probabiliti
                db.session.commit()
            else:
                # Jika email belum ada, buat entri baru dalam database RiwayatKlasifikasi
                new_riwayat_klasifikasi = RiwayatKlasifikasi(nama=nm, tanggal=tgl, email=email, umur=umr, jenis_kelamin=jk, image_path=file_path, klasifikasi_class=klasifikasi_class, probabiliti=probabiliti)
                db.session.add(new_riwayat_klasifikasi)
                db.session.commit()

        # Ambil semua riwayat klasifikasi dari database
        riwayat_klasifikasi = RiwayatKlasifikasi.query.all()
        
        comment = request.form.get('comment', '')

        # Preprocessing
        preprocessed_comment = preprocess_text(comment)

        # Tokenization
        tokenized_comment = word_tokenize(preprocessed_comment)

        # Stopword Removal
        stopwords_removed = remove_stopwords(tokenized_comment)

        # Slang Word
        path_dataslang = open("kamus kata baku 1.csv")
        dataslang = pd.read_csv(path_dataslang, encoding="utf-8", header=None, sep=";")

        formal_comment = [replace_slang(word, dataslang) for word in stopwords_removed]

        # Stemming
        factory = StemmerFactory()
        ind_stemmer = factory.create_stemmer()

        stemmed_comment = stemmer(formal_comment, ind_stemmer)

        # Load model and vectorizer
        file_path = "training.pickle"
        with open(file_path, "rb") as file:
            data_train = pickle.load(file)

        vectorizer = TfidfVectorizer()
        train_vector = vectorizer.fit_transform(data_train)

        load_model = joblib.load(open("hasil-sentimen.pkl", "rb"))

        # Predict sentiment
        result = predict_sentiment(comment, vectorizer, load_model)

        # Save the comment and sentiment to the database
        new_comment = Comment(comment=comment, sentiment=result)
        db.session.add(new_comment)
        db.session.commit()
        # Hitung jumlah sentimen untuk setiap kelas
        positive_count = len(Comment.query.filter_by(sentiment='positif').all())
        negative_count = len(Comment.query.filter_by(sentiment='negatif').all())
        netral_count = len(Comment.query.filter_by(sentiment='netral').all())
        total_count = positive_count + netral_count + negative_count
        comments = Comment.query.all()
        chart_path = create_sentiment_chart(positive_count, negative_count, netral_count)
        if request.referrer and "/dashboard" in request.referrer:
            return redirect('/dashboard')
        else:
            # Jika formulir dikirim dari root ("/"), tampilkan hasil di root ("/")
            # return redirect('/')

            return render_template('dashboard.html', chart_path=chart_path, total_count=total_count, comments=comments, positive_count=positive_count, negative_count=negative_count, netral_count=netral_count, klasifikasi_class=existing_klasifikasi.klasifikasi_class, probabiliti=existing_klasifikasi.probabiliti, image=existing_klasifikasi.image_path, nm=existing_klasifikasi.nama, tgl=existing_klasifikasi.tanggal, email=existing_klasifikasi.email, umr=existing_klasifikasi.umur, jk=existing_klasifikasi.jenis_kelamin, riwayat_klasifikasi=riwayat_klasifikasi)

    # Ambil semua riwayat klasifikasi dari database jika user sudah login
    if 'email' in session:
    # Ambil semua riwayat klasifikasi dari database
        riwayat_klasifikasi = RiwayatKlasifikasi.query.all()

        # Ambil semua komentar dan hasil sentimen dari database
        comments = Comment.query.all()

        # Hitung jumlah sentimen untuk setiap kelas
        positive_count = len(Comment.query.filter_by(sentiment='positif').all())
        negative_count = len(Comment.query.filter_by(sentiment='negatif').all())
        netral_count = len(Comment.query.filter_by(sentiment='netral').all())
        total_count = positive_count + netral_count + negative_count

        # Buat chart sentimen
        chart_path = create_sentiment_chart(positive_count, negative_count, netral_count)

        return render_template('dashboard.html', chart_path=chart_path, total_count=total_count, comments=comments, positive_count=positive_count, negative_count=negative_count, netral_count=netral_count, riwayat_klasifikasi=riwayat_klasifikasi)
    else:
        return redirect('/login')

@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.get_json()
    user_message = data["user_message"]
    
    # Panggil fungsi chatbot_response untuk mendapatkan jawaban
    response = chatbot_response(user_message)
    
    return ({"response": response})


        

@app.route("/get_klasifikasi", methods=['GET'])
def get_klasifikasi():
    data_klasifikasi = Klasifikasi.query.all()
    klasifikasi_list = [
        {
            'id': klasifikasi.id,
            'nama': klasifikasi.nama,
            'umur': klasifikasi.umur,
            'jenis_kelamin': klasifikasi.jenis_kelamin,
            'image_path': klasifikasi.image_path,
            'klasifikasi_class': klasifikasi.klasifikasi_class,
            'probabiliti': klasifikasi.probabiliti
        }
        for klasifikasi in data_klasifikasi
    ]
    return ({'klasifikasi': klasifikasi_list})