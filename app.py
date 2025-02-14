from flask import Flask, request, render_template, jsonify
from scripts.predict import predict_sentiment  # Your prediction function

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')




@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form.get('message', '')
        prediction = predict_sentiment(user_input)
        return jsonify({'prediction': prediction})
    return render_template('predict.html')  # Handles GET requests


if __name__ == '__main__':
    app.run(host="0.0.0.0" , port='5050')