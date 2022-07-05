from flask import Flask
app = Flask(__name__)
@app.route("/detect")
def hello():
    return "Hello, detector!"
if __name__ == "__main__":
    app.run()
