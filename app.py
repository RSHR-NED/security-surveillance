from flask import Flask, Response, flash, redirect, render_template, request, url_for
from camera_feed import VideoStream
import threading
from main import main

app = Flask(__name__)
app_interface = Flask(__name__)
app_face_recognizer = Flask(__name__)


@app_face_recognizer.route('/')
def index():
    return redirect(url_for('video'))


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


@app_interface.route('/')
def index():
    return render_template('add_faces.html', title='Add Faces')


@app_interface.route('/add_faces', methods=['POST', 'GET'])
def add_faces():
    if request.method == 'POST':
        flash('Face added POST (debug)')
        return redirect(url_for('index'))

    # Request method is GET
    return render_template('add_faces.html', title='Add Faces')

@app_interface.route("/mark_faces", methods=['POST', 'GET'])
def mark_faces():
    if request.method == 'POST':
        flash('Face marked POST (debug)')
        return redirect(url_for('index'))

    # Request method is GET
    return render_template('mark_faces.html', title='Mark Faces')



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