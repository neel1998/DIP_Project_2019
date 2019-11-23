from flask import Flask, render_template, request
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/upload1",methods=['POST'])
def upload1():
	filelist = request.files.getlist('file')
	print(filelist[0].path)
	return "got file"
if __name__ == "__main__":
	app.run(debug=True)