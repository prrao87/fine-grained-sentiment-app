import os
from flask import Flask, request, render_template
from lime_explainer import explainer, tokenizer, METHODS

app = Flask(__name__)
SECRET_KEY = os.urandom(24)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def explain():
    if request.method == 'POST':
        text = tokenizer(request.form['entry'])
        method = request.form['classifier']
        if not text:
            raise ValueError("Please enter a sentence with at least a few words.")

        exp = explainer(method,
                        path_to_file=METHODS[method]['file'],
                        text=text,
                        num_samples=1000)
        exp = exp.as_html()

        return render_template('result.html', exp=exp)
    return render_template('index.html')


if __name__ == '__main__':
    app.secret_key = SECRET_KEY
    app.run(debug=True)
