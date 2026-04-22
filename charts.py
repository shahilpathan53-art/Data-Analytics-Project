import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# dataset load
data = pd.read_csv("Admission_Predict_Ver1.1.csv")

# column names clean
data.columns = data.columns.str.strip()
data.columns = data.columns.str.replace(" ", "_")

# -------------------------------
# 1 GRE vs Chance (Scatter Plot)
# -------------------------------
plt.figure()
sns.scatterplot(x=data["GRE_Score"], y=data["Chance_of_Admit"])
plt.title("GRE Score vs Chance of Admit")
plt.savefig("static/gre_vs_chance.png")
plt.close()

# -------------------------------
# 2 CGPA vs University Rating (Bar)
# -------------------------------
plt.figure()
sns.barplot(x=data["University_Rating"], y=data["CGPA"])
plt.title("CGPA vs University Rating")
plt.savefig("static/cgpa_university_rating_bar.png")
plt.close()

# -------------------------------
# 3 University Rating vs Chance (Box)
# -------------------------------
plt.figure()
sns.boxplot(x=data["University_Rating"], y=data["Chance_of_Admit"])
plt.title("University Rating vs Chance of Admit")
plt.savefig("static/university_rating_boxplot.png")
plt.close()

# -------------------------------
# 4 Feature Relationship (Heatmap)
# -------------------------------
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("static/correlation_heatmap.png")
plt.close()

# -------------------------------
# 5 CGPA Distribution (Histogram)
# -------------------------------
plt.figure()
plt.hist(data["CGPA"], bins=20)
plt.title("CGPA Distribution")
plt.savefig("static/cgpa_histogram.png")
plt.close()

# -------------------------------
# 6 Research Distribution (Pie)
# -------------------------------
research_count = data["Research"].value_counts()

plt.figure()
plt.pie(research_count, labels=["Research","No Research"], autopct="%1.1f%%")
plt.title("Research Distribution")
plt.savefig("static/research_pie_chart.png")
plt.close()

print("✅ All charts saved in static folder")