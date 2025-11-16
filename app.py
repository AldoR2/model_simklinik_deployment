from flask import Flask

app = Flask(__name__)


@app.route('/')
def index():
    return {'status': 'success', 'message': 'Service is up'}, 200


app.run(debug=True)