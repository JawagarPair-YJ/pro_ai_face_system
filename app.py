
from flask import Flask, render_template, request, redirect, url_for, send_file
import sqlite3
import cv2
import os
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd

app = Flask(__name__)
DB = "attendance.db"

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("CREATE TABLE IF NOT EXISTS students(id INTEGER PRIMARY KEY,name TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS staffs(id INTEGER PRIMARY KEY,name TEXT)")
    c.execute('''CREATE TABLE IF NOT EXISTS attendance(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER,
        name TEXT,
        role TEXT,
        date TEXT,
        time TEXT)''')

    conn.commit()
    conn.close()

init_db()

def train_model(dataset_path, trainer_file):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = []
    ids = []

    for file in os.listdir(dataset_path):
        path = os.path.join(dataset_path, file)

        img = Image.open(path).convert('L')
        img_np = np.array(img, 'uint8')

        id = int(file.split(".")[1])

        detected = detector.detectMultiScale(img_np)

        for (x, y, w, h) in detected:
            faces.append(img_np[y:y+h, x:x+w])
            ids.append(id)

    if len(faces) > 0:
        recognizer.train(faces, np.array(ids))
        recognizer.save(trainer_file)

@app.route('/')
def home():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT * FROM attendance ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return render_template("index.html", attendance=data)

@app.route('/student_register', methods=['GET', 'POST'])
def student_register():
    if request.method == 'POST':
        student_id = request.form['id']
        name = request.form['name']

        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO students(id, name) VALUES (?, ?)", (student_id, name))
        conn.commit()
        conn.close()

        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        count = 0

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                count += 1
                cv2.imwrite(f"dataset/students/User.{student_id}.{count}.jpg", gray[y:y+h, x:x+w])
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

            cv2.imshow("Student Registration", img)

            if cv2.waitKey(1) == ord('q') or count >= 30:
                break

        cam.release()
        cv2.destroyAllWindows()

        train_model("dataset/students", "trainer/student_trainer.yml")

        return redirect(url_for('home'))

    return render_template("student_register.html")

@app.route('/staff_register', methods=['GET', 'POST'])
def staff_register():
    if request.method == 'POST':
        staff_id = request.form['id']
        name = request.form['name']

        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO staffs(id, name) VALUES (?, ?)", (staff_id, name))
        conn.commit()
        conn.close()

        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        count = 0

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                count += 1
                cv2.imwrite(f"dataset/staffs/User.{staff_id}.{count}.jpg", gray[y:y+h, x:x+w])
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

            cv2.imshow("Staff Registration", img)

            if cv2.waitKey(1) == ord('q') or count >= 30:
                break

        cam.release()
        cv2.destroyAllWindows()

        train_model("dataset/staffs", "trainer/staff_trainer.yml")

        return redirect(url_for('home'))

    return render_template("staff_register.html")

def mark_attendance(role, trainer_file, table_name):

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_file)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    conn = sqlite3.connect(DB)
    c = conn.cursor()

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for(x, y, w, h) in faces:

            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 70:
                c.execute(f"SELECT name FROM {table_name} WHERE id=?", (id,))
                result = c.fetchone()

                if result:
                    name = result[0]

                    now = datetime.now()
                    date = now.strftime("%Y-%m-%d")
                    time = now.strftime("%H:%M:%S")

                    c.execute("SELECT * FROM attendance WHERE person_id=? AND role=? AND date=?", (id, role, date))
                    existing = c.fetchone()

                    if not existing:
                        c.execute("INSERT INTO attendance(person_id, name, role, date, time) VALUES (?, ?, ?, ?, ?)",
                                  (id, name, role, date, time))
                        conn.commit()

                    cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        cv2.imshow(role + " Attendance", img)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    conn.close()

@app.route('/student_attendance')
def student_attendance():
    mark_attendance("Student", "trainer/student_trainer.yml", "students")
    return redirect(url_for('home'))

@app.route('/staff_attendance')
def staff_attendance():
    mark_attendance("Staff", "trainer/staff_trainer.yml", "staffs")
    return redirect(url_for('home'))

@app.route('/download')
def download():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM attendance", conn)
    file = "attendance_report.csv"
    df.to_csv(file, index=False)
    conn.close()
    return send_file(file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
