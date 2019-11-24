from flask import Flask, render_template, request, redirect
import os

image_path = './upload_images/'
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('index.html',files=[])

@app.route("/upload1",methods=['POST'])
def upload1():
	filelist = request.files.getlist('file')
	file = filelist[0]
	fn = file.filename
	print(fn)
	file.save(image_path+fn)
	return render_template('index.html',files=[fn])
if __name__ == "__main__":
	app.run(debug=True)