from flask import Flask, jsonify
import json
import perceptron
app = Flask(__name__)


@app.route('/health')
def check_health():
    return 'Server healthy'


@app.route("/predict", methods=["GET"])
def compute():
    mlp = perceptron.MLP([3136, 1568, 784, 392, 196, 56, 3])
    mlp.load("the best parameters save")
    post_inputs = []
    sail_class = mlp.predict(post_inputs)
    if sail_class[0] == 1:
        predicted = "sailboat"
    elif sail_class[1] == 1:
        predicted = "windsurf"
    else:
        predicted = "jetski"
    res = {
        "class_predicted": predicted
    }
    return res


if __name__ == '__main__':
    app.run()
