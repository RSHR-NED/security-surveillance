import threading
from flask import Flask, render_template, Response
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
    return Response(generate(video_thread.join()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # start video stream thread
    video_stream = VideoStream()
    video_thread = threading.Thread(target=generate, args=(video_stream,))
    video_thread.start()
    app.run(debug=True)
