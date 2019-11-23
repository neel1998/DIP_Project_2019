from flask import Flask, render_template
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/upload1",methods=['POST'])
def upload1():
	return "got file"
if __name__ == "__main__":
	app.run(debug=True)