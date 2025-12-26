from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)   # 🔥 VERY IMPORTANT

model = pickle.load(open("email_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    email_text = data["email"]

    vect = vectorizer.transform([email_text])
    pred = model.predict(vect)[0]
    confidence = float(max(model.predict_proba(vect)[0]))

    return jsonify({
        "success": True,
        "category": pred,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
