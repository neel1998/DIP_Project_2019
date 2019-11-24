from flask import Flask, render_template, request, redirect
import os
import sys
sys.path.insert(1, '../')
import patchmatch_object_removal

image_path = './static/upload_images/'
app = Flask(__name__)
upload_images = ['','']

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
	fn = image_path + fn
	upload_images[0] = fn
	return render_template('index.html',files=upload_images, res='')

@app.route("/upload2",methods=['POST'])
def upload2():
	filelist = request.files.getlist('file')
	file = filelist[0]
	fn = file.filename
	print(fn)
	file.save(image_path+fn)
	fn = image_path + fn
	upload_images[1] = fn
	return render_template('index.html',files=upload_images,res='')

@app.route("/get_results")
def get_resutls():
	print(upload_images)
	os.system("python3 ../patchmatch_object_removal.py " + upload_images[0] + " " + upload_images[1] + " 41 3")
	res_name = './static/results/' + upload_images[0].split("/")[-1].split(".")[0] + "_" +  upload_images[1].split("/")[-1].split(".")[0] + "_res.jpg"
	return render_template('index.html',files=upload_images,res=res_name)	

if __name__ == "__main__":
	app.run(debug=True)