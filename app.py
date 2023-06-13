import cv2
import sys
import numpy as np
from PIL import Image
import os
from flask import Flask, render_template, request, Response, jsonify, redirect, url_for,session,flash
from flask_mysqldb import MySQL
import base64
import datetime
import face_recognition
import csv
import glob
from datetime import date
import uuid

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
        #print("Error: Failed to open video capture device")
        return

    count = 0
    max_capture_attempts = 5
    capture_attempts = 0

    # Create the new folder within the "dataset" folder
    folder_path = os.path.join('dataset', year, department, classr, admissionNo)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        #print("Folder created successfully.")


    while count < 15:
        ret, frame = video_capture.read()

        if not ret:
            capture_attempts += 1

            if capture_attempts > max_capture_attempts:
                #print("Error: Maximum capture attempts reached")
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
    return render_template('admin/adminLogin.html')

@app.route('/trainfaces')
def train_faces():
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
        #print("Face trained with name =", request.form.get('name')," and admission number = ",request.form.get('admission-no'))
        return render_template('admin/train.html')

@app.route('/teacherSelectClass')
def teacherSelectClass():
    # Render the teacher dashboard template
    return render_template('teacher/teacherSelectClass.html')

@app.route('/logout-teacher', methods = ['POST'])
def logoutTeacher():
    # Clear the session
    session.clear()
    return render_template('teacher/teacherLogin.html')

#RECOGNITION CODE
@app.route('/capture', methods=['POST'])
def capture():
    # Get the captured photo from the request data
    captured_photo = request.form['photo']
    department = request.form.get('department')
    year = request.form.get('year')
    class_ = request.form.get('class')
    session['department'] = department
    session['year'] = year
    session['class'] = class_
    
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
    # test_image_path = "20230610_140123.jpg"
    # test_image_path = "20230610_141433.jpg"
    department = request.args.get('department')
    year = request.args.get('year')
    class_ = request.args.get('class_')
    
    dataset_folder = 'dataset/' + year + "/" + department + "/" + department + " " +class_
    #print(dataset_folder)

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

    recognized_names = []

    for face_encoding in face_encodings:
        print("===============  FACE OBTAINED =============")
        # Compare the face encoding with the known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = 'Unknown'

        if face_encodings:
            encoding = face_encoding
        else:
            # Handle the case when no face is detected in the image
            print("No face detected in the image")
            continue

        if True in matches:
            matched_indexes = [i for i, match in enumerate(matches) if match]
            face_distances = face_recognition.face_distance(known_face_encodings, encoding)
            best_match_index = min(range(len(face_distances)), key=face_distances.__getitem__)
            name = known_face_names[best_match_index]

        # Remove dot and following number from the name
        name_parts = name.rsplit('.', 1)
        name = name_parts[0]

        # print the recognized face name
        print("Recognized face:", name)

        recognized_names.append(name)

    print("Recognized names:", recognized_names)
    absent_names = tuple({name.rsplit('.', 1)[0] for name in known_face_names} - set(recognized_names))

    print("Missing names:", absent_names)


    return render_template('teacher/face_rec_and_mark_attend.html', recognized_names=recognized_names,absent_names = absent_names)

    

@app.route('/photo/<file_name>')
def show_photo(file_name):
    # Build the file path of the captured photo
    photo_path = file_name

    # Render the template to display the captured photo
    return render_template('teacher/face_rec_and_mark_attend.html', photo_path=photo_path)

@app.route('/capturephoto')
def capturePhotoNav():
    return render_template('teacher/imageCapture.html')

@app.route('/add-teacher')
def add_teacher():
    # Render the add_teacher template
    return render_template('admin/add_teacher.html')

@app.route('/navigateToViewStudents')
def navigateToViewStudents():
    if 'username' not in session:
        return render_template('teacher/teacherLogin.html')
    att_id = session.get('att_id')
    print('attendance id =',att_id)
    student_details = []
    with mysql.connection.cursor() as cur:
        query = "SELECT students.ad_no, students.name, attendance.att_status FROM students INNER JOIN attendance ON students.ad_no = attendance.ad_no WHERE attendance.att_id = %s;"
        cur.execute(query, (att_id,))
        rows = cur.fetchall()
        for row in rows:
            ad_no = row[0]
            student_name = row[1]
            attendance_status = "Present" if row[2] else "Absent"
            student_details.append({"ad_no": ad_no, "student_name": student_name, "attendance_status": attendance_status})

    return render_template('teacher/viewAttendance.html', attendances=student_details)

@app.route('/navigateToDashboard')
def navigateToDashboard():
    if 'username' not in session:
        return render_template('teacher/teacherLogin.html')

    username = session.get('username')
    teacherName = session.get('teacherName')

    cursor = mysql.connection.cursor()

    if 'att_id' in session:
        att_id = session.get('att_id')
        query = "SELECT COUNT(*) AS total_count, SUM(CASE WHEN att_status = 'Absent' THEN 1 ELSE 0 END) AS absent_count, (SUM(CASE WHEN att_status = 'Present' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS present_percentage FROM attendance WHERE att_id = %s"
        cursor.execute(query, (att_id,))

        result = cursor.fetchone()

        absent_count = result[1]
        present_percentage = result[2]
        present_percentage = int(present_percentage)
        # print('workingggggg')
    else:
        absent_count = 0
        present_percentage = 0

    cursor.close()

    return render_template('teacher/teacherDashboard.html', username=username, teacherName=teacherName,
                           absent_count=absent_count, present_percentage=present_percentage)

####################################################################################################################################
                                                                # DATABASE


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

  

#   LOGIN SECTION TEACHER
# -----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT username,name FROM teachers WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone()

        if user:
            session['username'] = user[0]
            session['teacherName'] = user[1]
            #print("user = ", user[1])
            return redirect('/navigateToDashboard')
        else:
            error = 'Invalid credentials. Please try again.'
            return render_template('login.html', error=error)
    
    return render_template('login.html')

#       Register names(attendance)
@app.route('/register', methods=['POST'])
def register():
    names = request.form.getlist('name')  # Get the list of recognized names from the form
    return "Name registered successfully!"


# admin -> section

# login
@app.route('/admin-login', methods=['GET', 'POST'])
def adminlogin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == 'admin' and password == 'root':
            return redirect(url_for('admin_dashboard'))
        else:
            error = 'Invalid username or password. Please try again.'
            return render_template('admin/adminLogin.html', error=error)

    return render_template('adminDashboard.html')

@app.route('/admin-dashboard')
def admin_dashboard():
    
    cursor = mysql.connection.cursor()
    # Fetch data from the database
    cursor.execute("SELECT COUNT(*) FROM students")
    total_students = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM teachers")
    total_teachers = cursor.fetchone()[0]

    # cursor.execute("SELECT COUNT(*) FROM staffs")
    total_staffs = 12

    # Pass the data to the template
    return render_template('admin/adminDashboard.html', total_students=total_students, total_teachers=total_teachers, total_staffs=total_staffs)

@app.route('/addTeacher', methods=['POST'])
def addTeacher():
    name = request.form['name']
    username = request.form['username']
    password = request.form['password']
    major = request.form['major']

    # Create a cursor to interact with the database
    cursor = mysql.connection.cursor()

    # Execute the SQL query to insert the new teacher
    query = "INSERT INTO teachers (name, username, password, major) VALUES (%s, %s, %s, %s)"
    values = (name, username, password, major)
    cursor.execute(query, values)

    # Commit the transaction to save the changes
    mysql.connection.commit()

    # Close the cursor
    cursor.close()

    # Return a JSON response with the success message
    return jsonify({'message': 'Teacher added successfully!'})

# Route for viewing teachers
@app.route('/view-teachers')
def view_teachers():
    # Query to get all teachers
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM teachers")
    teachers = cursor.fetchall()
    
    # Render the view_teachers template with the teacher data
    return render_template('admin/view_teachers.html', teachers=teachers)

@app.route('/doneViewing', methods=['POST'])
def doneViewing():
    if request.method == 'POST':
        dept_id, teach_id, class_id = "", "", ""
        names = request.form.getlist('name')
        absent_names = request.form.getlist('absent_name')
        dept = session.get('department')
        class_ = dept + " " + session.get('class')
        year = ""
        if(session.get('year') == "First Year"):
            year = "1"
        elif(session.get('year') == "Second Year"):
            year = "2"
        elif(session.get('year') == "Third Year"):
            year = "3"
        else:
            year = "4"
        username = session.get('username')
        current_date = date.today()
        studentsAdNo = []

        with mysql.connection.cursor() as cur:
            # Get department ID and class ID
            query = "SELECT dept_id, cl_id FROM department WHERE year = %s AND name = %s AND className = %s;"
            cur.execute(query, (year, dept, class_))
            row = cur.fetchone()
            if row:
                dept_id = row[0]
                class_id = row[1]
            else:
                return "No such department"

            # Get teacher ID
            query = "SELECT teacher_id FROM teachers WHERE username = %s;"
            cur.execute(query, (username,))
            row = cur.fetchone()
            if row:
                teach_id = row[0]
            else:
                return "No such teacher"

            # Insert data into attendance_records table
            query = "INSERT INTO attendance_record (date, year, dept_id, cl_id, teacher_id) VALUES (%s, %s, %s, %s, %s);"
            cur.execute(query, (current_date, year, dept_id, class_id, teach_id))

            att_id = cur.lastrowid  # Get the auto-incremented attendance ID
            session['att_id'] = att_id
            print("att_id = ",att_id)
            # Insert data into attendance table for present students
            for name in names:
                #print("student name = ", name)
                query = "SELECT ad_no FROM students WHERE name = %s AND dept_id = %s AND year = %s;"
                cur.execute(query, (name, dept_id, year))
                row = cur.fetchone()
                if row:
                    ad_no = row[0]
                    studentsAdNo.append(ad_no)

                    att_status = True  # Present
                    query = "INSERT INTO attendance (att_id, dept_tid, cl_id, year, ad_no, att_status) VALUES (%s, %s, %s, %s, %s, %s);"
                    cur.execute(query, (att_id, dept_id, class_id, year, ad_no, att_status))

            # Insert data into attendance table for absent students
            for name in absent_names:
                #print("absent student name = ", name)
                query = "SELECT ad_no FROM students WHERE name = %s AND dept_id = %s AND year = %s;"
                cur.execute(query, (name, dept_id, year))
                row = cur.fetchone()
                if row:
                    ad_no = row[0]
                    studentsAdNo.append(ad_no)

                    att_status = False  # Absent
                    query = "INSERT INTO attendance (att_id, dept_tid, cl_id, year, ad_no, att_status) VALUES (%s, %s, %s, %s, %s, %s);"
                    cur.execute(query, (att_id, dept_id, class_id, year, ad_no, att_status))

            mysql.connection.commit()

        # username = session.get('username')
        #print("students ad_no =", studentsAdNo)
        return redirect(url_for('navigateToDashboard'))

    return "Invalid request method"

@app.route('/updateAttendance', methods=['POST'])
def updateAttendance():
    if request.method == 'POST':
        ad_no = request.form.get('ad_no')
        #print(ad_no)
        attendance_status = request.form.get('attendance_status')

        # Update the attendance status based on the current status
        if attendance_status == 'Present':
            new_status = '0'
        else:
            new_status = '1'

        # Perform the update operation in the database
        with mysql.connection.cursor() as cur:
            query = "UPDATE attendance SET att_status = %s WHERE ad_no = %s;"
            cur.execute(query, (new_status, ad_no))
            mysql.connection.commit()

        flash('Attendance status updated successfully.')
        return redirect(url_for('navigateToViewStudents'))

    return "Invalid request method"




if __name__ == '__main__':
    app.run(debug=True) 