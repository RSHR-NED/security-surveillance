from flask import Flask, Response, flash, redirect, render_template, request, url_for
from camera_feed import VideoStream


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', title='Camera Feed')


def generate(stream):
    while True:
        frame = stream.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type:  image/jpeg\r\n\r\n' + frame +
              b'\r\n\r\n')


@app.route('/video')
def video():
    return Response(generate(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/add_faces', methods=['POST', 'GET'])
def add_faces():
    if request.method == 'POST':
        flash('Face added POST (debug)')
        return redirect(url_for('index'))

    # Request method is GET
    return render_template('add_faces.html', title='Add Faces')

@app.route("/mark_faces", methods=['POST', 'GET'])
def mark_faces():
    if request.method == 'POST':
        flash('Face marked POST (debug)')
        return redirect(url_for('index'))

    # Request method is GET
    return render_template('mark_faces.html', title='Mark Faces')
    
if __name__ == '__main__':
    # start video stream thread
    video_stream = VideoStream()
    app.run(debug=True)