import cv2
import sys
import numpy as np
from PIL import Image
import os
from flask import Flask, render_template, request, Response, jsonify, redirect, url_for,session
from flask_mysqldb import MySQL
import base64
import datetime
import face_recognition
import csv
import glob

face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
count = 0

app = Flask(__name__)
# init_db(app)

def generate_id():
    existing_ids = [int(file.split(".")[2]) for file in os.listdir("dataset") if file.endswith(".jpg") and len(file.split(".")) >= 3]
    new_id = max(existing_ids) + 1 if existing_ids else 1
    return new_id

def train_face(name,admissionNo,age,gender,department,year,classr):
    video_capture = cv2.VideoCapture(0)  # Adjust the video source index if needed
    
    # Check if the video capture device is opened successfully
    if not video_capture.isOpened():
        print("Error: Failed to open video capture device")
        return

    count = 0
    max_capture_attempts = 5
    capture_attempts = 0

    # Create the new folder within the "dataset" folder
    folder_path = os.path.join('dataset', year, department, classr, admissionNo)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder created successfully.")
    else:
        print("Folder already exists.")


    while count < 15:
        ret, frame = video_capture.read()

        if not ret:
            capture_attempts += 1

            if capture_attempts > max_capture_attempts:
                print("Error: Maximum capture attempts reached")
                break

            continue

        capture_attempts = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.FONT_HERSHEY_SIMPLEX
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1

            cv2.imwrite(f"dataset/{year}/{department}/{classr}/{admissionNo}/{name}.{count}.jpg", gray[y:y + h, x:x + w])

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(100) == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()

    if count == 15:
        return 'Training complete'
    else:
        return 'Training incomplete'


@app.errorhandler(500)
def handle_internal_server_error(e):
    # Handle the internal server error here
    return "Internal Server Error", 500


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/mark_attendance', methods=['GET', 'POST'])
def markAttendance():
    # return render_template('teacher/teacherDashboard.html')
    return render_template('teacher/teacherLogin.html')


@app.route('/train_page', methods=['GET', 'POST'])
def train_page():
    return render_template('admin/train.html')


@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        name = request.form.get('name')
        admissionNo = request.form.get('admission-no')
        age = request.form.get('age')
        gender = request.form.get('gender')
        department = request.form.get('department')
        year = request.form.get('year')
        classr = request.form.get('division')
        id = train_face(name,admissionNo,age,gender,department,year,classr)
        print("Face trained with name =", request.form.get('name')," and admission number = ",request.form.get('admission-no'))
        return render_template('admin/train.html')

@app.route('/teacherDashboard')
def teacherDashboard():
    # Render the teacher dashboard template
    return render_template('teacher/teacherDashboard.html')


#RECOGNITION CODE
@app.route('/capture', methods=['POST'])
def capture():
    # Get the captured photo from the request data
    captured_photo = request.form['photo']
    department = request.form.get('department')
    year = request.form.get('year')
    class_ = request.form.get('class')
    
    # Decode the base64-encoded photo data
    photo_data = base64.b64decode(captured_photo.split(',')[1])

    # Generate a unique file name with the current date and time
    now = datetime.datetime.now()
    file_name = now.strftime("%Y%m%d_%H%M%S") + '.jpg'

    # Save the photo to a file
    photo_path = 'images/' + file_name
    with open(photo_path, 'wb') as file:
        file.write(photo_data)

    # Redirect to the page showing the captured photo
    return redirect(url_for('recognize', file_name=file_name, department=department, year=year, class_=class_))

@app.route('/recognize')
def recognize():
    test_image_path = request.args.get('file_name')
    department = request.args.get('department')
    year = request.args.get('year')
    class_ = request.args.get('class_')
    
    dataset_folder = 'dataset/' + year + "/" + department + "/" + class_
    print(dataset_folder)

    # Load the known face images and their corresponding names from the dataset
    known_face_encodings = []
    known_face_names = []

    for root, dirs, files in os.walk(dataset_folder):
        for file_name in files:
            image_path = os.path.join(root, file_name)
            name = os.path.splitext(file_name)[0]

            # Load the image and compute the face encoding
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(name)
            else:
                # Handle the case when no face is detected in the image
                print("No face detected in the image:", image_path)

    # Load the test image for recognition
    test_image = face_recognition.load_image_file('Images/' + test_image_path)

    # Find faces in the test image
    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)

    for face_encoding in face_encodings:
        print("===============im here=============")
        # Compare the face encoding with the known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = 'Unknown'

        if face_encodings:
            encoding = face_encodings[0]
        else:
            # Handle the case when no face is detected in the image
            print("No face detected in the image")
            continue

        if True in matches:
            matched_indexes = [i for i, match in enumerate(matches) if match]
            face_distances = face_recognition.face_distance(known_face_encodings, encoding)
            best_match_index = min(range(len(face_distances)), key=face_distances.__getitem__)
            name = known_face_names[best_match_index]

        # Print the recognized face name
        print("Recognized face:", name)
    return render_template('teacher/face_rec_and_mark_attend.html', recognized_name=name)
    

@app.route('/photo/<file_name>')
def show_photo(file_name):
    # Build the file path of the captured photo
    photo_path = file_name

    # Render the template to display the captured photo
    return render_template('teacher/face_rec_and_mark_attend.html', photo_path=photo_path)

@app.route('/capturephoto')
def capturePhotoNav():
    return render_template('teacher/imageCapture.html')
#   LOGIN SECTION TEACHER
# -----------------

app.secret_key = '1222'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'ipts'
mysql = MySQL(app)

# Test connection
@app.route('/testconnection')
def test_connection():
    try:
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT 1')
        result = cursor.fetchone()
        return f"Database connection successful. Result: {result[0]}"
    except Exception as e:
        return f"Database connection failed. Error: {str(e)}"
    

# Login section
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM teachers WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone()

        if user:
            session['username'] = user[1]
            return redirect('/teacherDashboard')
        else:
            error = 'Invalid credentials. Please try again.'
            return render_template('login.html', error=error)
    
    return render_template('login.html')



if __name__ == '__main__':
    app.run(debug=True)
