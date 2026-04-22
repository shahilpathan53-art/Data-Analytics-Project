import os
import csv


from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("admission_model.pkl")


# ---------------- HOME DASHBOARD ----------------

@app.route("/")
def home():
    return render_template("home.html")


# ---------------- PREDICTION PAGE ----------------

@app.route("/predict_page")
def predict_page():
    return render_template("index.html")


# ---------------- CHART DASHBOARD ----------------

@app.route("/charts")
def charts():
    return render_template("charts.html")


# ---------------- ABOUT PAGE ----------------

@app.route('/about')
def about():
    return render_template('about.html')

# ---------------- DATASET CHARTS ----------------

# 1 GRE vs Chance (Scatter Plot)
@app.route("/gre_chart")
def gre_chart():
    return render_template("gre_chart.html")


# 2 CGPA vs University Rating (Bar Chart)
@app.route("/cgpa_chart")
def cgpa_chart():
    return render_template("cgpa_chart.html")


# 3 University Rating vs Chance (Box Plot)
@app.route("/rating_chart")
def rating_chart():
    return render_template("rating_chart.html")


# 4 Feature Relationship Heatmap
@app.route("/heatmap_chart")
def heatmap_chart():
    return render_template("heatmap_chart.html")


# 5 CGPA Distribution (Histogram)
@app.route("/cgpa_distribution")
def cgpa_distribution():
    return render_template("cgpa_distribution.html")


# 6 Research Distribution
@app.route("/research_chart")
def research_chart():
    return render_template("research_chart.html")

# ---------------- DATASET  ----------------


@app.route('/dataset')
def dataset():
    data = []
    csv_path = os.path.join(os.path.dirname(__file__), 'Admission_Predict_Ver1.1.csv')
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append([
                int(row['Serial No']),
                int(row['GRE Score']),
                int(row['TOEFL Score']),
                int(row['University Rating']),
                float(row['SOP']),
                float(row['LOR ']),        # note: space after LOR
                float(row['CGPA']),
                int(row['Research']),
                float(row['Chance of Admit '])  # note: space after Admit
            ])
    return render_template('dataset.html', dataset=data)

# ---------------- MODEL INFORMATION ----------------

@app.route('/model')
def model_info():
    return render_template('model.html')


# ---------------- PREDICTION LOGIC ----------------

@app.route("/predict", methods=["POST"])
def predict():

    gre = float(request.form["gre"])
    toefl = float(request.form["toefl"])
    rating = float(request.form["rating"])
    sop = float(request.form["sop"])
    lor = float(request.form["lor"])
    cgpa = float(request.form["cgpa"])
    research = float(request.form["research"])

    input_data = np.array([[gre, toefl, rating, sop, lor, cgpa, research]])

    chance = model.predict(input_data)[0]

    percentage = round(chance * 100, 2)

    # Admission Classification
    if percentage >= 80:
        admission_status = "🎓 High Chance of Admission"
    elif percentage >= 50:
        admission_status = "⚖️ Moderate Chance of Admission"
    else:
        admission_status = "⚠️ Low Chance of Admission"

    return render_template(
        "index.html",
        prediction=percentage,
        status=admission_status
    )


# ---------------- RUN APP ----------------

if __name__ == "__main__":
    app.run(debug=True)