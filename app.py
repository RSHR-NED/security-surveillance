import threading
from flask import Flask, Response, flash, redirect, render_template, request, url_for
from camera_feed import VideoStream
import os
import secrets
from PIL import Image
from helpers import FaceRecognizer
import face_recognition

# app defined two flask app
app = Flask(__name__, static_folder = 'unidentified_faces', template_folder='templates')
app_interface = Flask(__name__, static_folder = 'unidentified_faces')
app_face_recognizer = Flask(__name__)

app_interface.config['SECRET_KEY'] = 'asdcsdsfcsdfsfs'
app_face_recognizer.config['SECRET_KEY'] = 'asdcsdsfcsdfsfs'
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
fc = FaceRecognizer()
def save_media(picture,name):# open s3 instance


    random_hex = secrets.token_hex(10)
    _, f_ext = os.path.splitext(picture.filename)
    pic_fname = name + f_ext
    print(pic_fname)
    pic_path = os.path.join(app.root_path, 'identified_faces', pic_fname)
    
    i = Image.open(picture)
    # resize and save
    bwidth = 600
    ratio = bwidth / float(i.size[0])
    height = int((float(i.size[1]) * ratio))
    i = i.resize((bwidth, height), Image.ANTIALIAS)
    i.save(pic_path)

    return pic_fname

@app_face_recognizer.route('/')
def index():
    return redirect(url_for('video'))
    # return render_template('index.html', title='Camera Feed')


def generate(stream):
    while True:
        frame = stream.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type:  image/jpeg\r\n\r\n' + frame +
              b'\r\n\r\n')


@app_face_recognizer.route('/video')
def video():
    return Response(generate(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

''' Here we add images and store their encodings to the identified.json data   '''
# @app_interface.route('/add_faces', methods=['POST', 'GET'])
@app_interface.route('/', methods=['POST', 'GET'])
def add_faces():
    if request.method == 'POST':
        name = request.form['name']
        img = request.files['image']
        label = request.form['label']
        if label == 'safe':
            label = True
        else:
            label = False
        file_name = save_media(img,name)
        # print(file_name)
        if file_name:
            image_array = face_recognition.load_image_file(('./identified_faces/' + file_name))
            face_encoding = face_recognition.face_encodings(image_array)[0] 
        identified_face_enc = fc.load_face_encodings()[0] 
        # get the dictionary of identified faces
        new_id = list(identified_face_enc.keys())[-1] + 1
        if face_encoding[0]:
            identified_face_enc[new_id] = [name,list(face_encoding),label]
            # print(identified_face_enc)
            fc.save_encodings(identified_face_enc,"./identified_faces_encodings.json")
                    
            # print('face_encode',face_encoding)
            flash('Image added successfully', 'success')
            return redirect(url_for('add_faces'))
    # Request method is GET
    return render_template('add_faces.html', title='Add Faces')

''' Here we display all the iimages from unidentified faces folder'''
# @app.route("/mark_faces", methods=['POST', 'GET'])
@app_interface.route("/mark_faces", methods=['POST', 'GET'])
def mark_faces():
    if request.method == 'POST':
        flash('Face marked POST (debug)')
        return redirect(url_for('mark_faces'))
    files_url = []
    for filename in os.listdir('./unidentified_faces'):
        files_url.append(filename)
    # Request method is GET
    return render_template('mark_faces.html', title='Mark Faces', files_url=files_url)
    
''' Remove the chosen file from the unidentified faces folder and store the encodings in identified.json'''
@app_interface.route("/mark_img", methods=['POST', 'GET'])
def mark_img():
    name = False
    # here we et the details of image marks as --- safe/unsafe
    if request.method == 'POST':
        flash('Face marked POST (debug)')
        name = request.form['name']
        label = request.form['label']
    file_name = request.args.get('search')
    # here we load encoding of file save it in identified.json and remove it from unidentified faces
    if file_name:
            image_array = face_recognition.load_image_file(('./unidentified_faces/' + file_name))
            face_encoding = face_recognition.face_encodings(image_array)[0] 
    # load_face_encodings returns a tuple of two dictionaries (identified, unidentified)
    identified_face_enc = fc.load_face_encodings()[0] 
    # get the dictionary of identified faces
    new_id = list(identified_face_enc.keys())[-1] + 1
    if name:
        identified_face_enc[new_id] = [name,list(face_encoding),label]
        fc.save_encodings(identified_face_enc,"./identified_faces_encodings.json")
        os.remove('./unidentified_faces/' + file_name)
        flash('Marked Successfully', 'success')
        return redirect(url_for('mark_faces'))

    return render_template('mark_img.html', title='Mark Faces', img=file_name)


# if __name__ == '__main__':
#     # start video stream thread
#     video_stream = VideoStream()
#     app.run(debug=True)
# With Multi-Threading Apps, YOU CANNOT USE DEBUG!
# Though you can sub-thread.
def run_app_interface():
    app_interface.run(host='127.0.0.1', port=5000, debug=False, threaded=True)

def run_app_face_recognizer():
    app_face_recognizer.run(host='127.0.0.1', port=5001, debug=False, threaded=True)


if __name__ == '__main__':
    # start video stream thread
    video_stream = VideoStream()
    # Executing the Threads seperatly.
    t1 = threading.Thread(target=run_app_interface)
    t2 = threading.Thread(target=run_app_face_recognizer)
    t1.start()
    t2.start()