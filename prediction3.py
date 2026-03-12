import os
import pandas as pd
import customtkinter as ctk
from tkinter import filedialog
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas

import nltk
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

from pyresparser import ResumeParser

# --------------------------
# UI SETTINGS
# --------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# --------------------------
# MACHINE LEARNING MODEL
# --------------------------
class PersonalityModel:

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.feature_names = []

    def train(self):
        data = pd.read_csv("personality_dataset.csv")

        # Convert all object columns to numeric
        for col in data.columns[:-1]:
            if data[col].dtype == object:
                # Map Yes/No to 1/0
                data[col] = data[col].map({"Yes": 1, "No": 0}).fillna(data[col])
                # Factorize any remaining strings
                if data[col].dtype == object:
                    data[col], _ = pd.factorize(data[col])

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        self.feature_names = list(X.columns)

        self.model.fit(X, y)

    def predict(self, values):
        converted = []
        try:
            for v in values:
                if isinstance(v, str):
                    if v.lower() == "yes":
                        converted.append(1)
                    elif v.lower() == "no":
                        converted.append(0)
                    else:
                        converted.append(float(v))
                else:
                    converted.append(float(v))
            return self.model.predict([converted])[0]
        except:
            return "Invalid Input"

# Initialize model and train
model = PersonalityModel()
model.train()

# --------------------------
# RESUME PARSING
# --------------------------
def parse_resume(path):
    try:
        data = ResumeParser(path).get_extracted_data()
        if "name" in data:
            del data["name"]
        return data
    except:
        return {}

# --------------------------
# FILE BROWSER
# --------------------------
def browse_file():
    filename = filedialog.askopenfilename(
        filetypes=[("PDF files", "*.pdf"), ("Word files", "*.docx")]
    )
    if filename:
        resume_path.set(filename)
        file_label.configure(text=os.path.basename(filename))

# --------------------------
# CHART DISPLAY
# --------------------------
def show_chart(values):
    labels = model.feature_names
    plt.figure(figsize=(6,4))
    plt.bar(labels, values, color='skyblue')
    plt.xticks(rotation=45)
    plt.title("Personality Feature Scores")
    plt.tight_layout()
    plt.show()

# --------------------------
# PDF REPORT
# --------------------------
def export_pdf(name, personality):
    c = canvas.Canvas("prediction_report.pdf")
    c.drawString(100,750,"Personality Prediction Report")
    c.drawString(100,720,f"Name: {name}")
    c.drawString(100,700,f"Predicted Personality: {personality}")
    c.save()

# --------------------------
# PREDICTION
# --------------------------
def predict_personality():
    name = name_entry.get()
    age = age_entry.get()

    if name == "" or age == "":
        result_label.configure(text="⚠ Enter Name and Age")
        return

    features = []
    numeric_features = []

    for fname, entry in feature_entries.items():
        val = entry.get().strip()
        if val == "":
            result_label.configure(text=f"⚠ Enter value for {fname}")
            return
        features.append(val)
        try:
            numeric_features.append(float(val if val.lower() not in ["yes","no"] else (1 if val.lower()=="yes" else 0)))
        except:
            numeric_features.append(0)

    personality = model.predict(features)
    result_label.configure(text=f"Predicted Personality: {personality}")

    # Show chart
    show_chart(numeric_features)

    # Generate PDF
    export_pdf(name, personality)

    # Show parsed resume
    if resume_path.get() != "":
        data = parse_resume(resume_path.get())
        resume_text.delete("1.0","end")
        for k,v in data.items():
            resume_text.insert("end",f"{k} : {v}\n")

# --------------------------
# MAIN WINDOW
# --------------------------
app = ctk.CTk()
app.title("AI Personality Prediction System")
app.geometry("900x700")

title = ctk.CTkLabel(app, text="AI Personality Prediction System", font=("Arial",30,"bold"))
title.pack(pady=20)

main_frame = ctk.CTkFrame(app)
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

resume_path = ctk.StringVar()

# LEFT PANEL
left_frame = ctk.CTkFrame(main_frame)
left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

ctk.CTkLabel(left_frame,text="Name").pack()
name_entry = ctk.CTkEntry(left_frame,width=250)
name_entry.pack(pady=5)

ctk.CTkLabel(left_frame,text="Age").pack()
age_entry = ctk.CTkEntry(left_frame,width=250)
age_entry.pack(pady=5)

ctk.CTkButton(left_frame,text="Upload Resume", command=browse_file).pack(pady=10)
file_label = ctk.CTkLabel(left_frame,text="No file selected")
file_label.pack()

# Features dynamically from dataset
feature_entries = {}
for fname in model.feature_names:
    ctk.CTkLabel(left_frame,text=fname).pack(pady=3)
    entry = ctk.CTkEntry(left_frame,width=250)
    entry.pack()
    feature_entries[fname] = entry

ctk.CTkButton(left_frame,text="Predict Personality", command=predict_personality).pack(pady=20)

result_label = ctk.CTkLabel(left_frame,text="Prediction will appear here", font=("Arial",16))
result_label.pack()

# RIGHT PANEL
right_frame = ctk.CTkFrame(main_frame)
right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

ctk.CTkLabel(right_frame,text="Parsed Resume Data", font=("Arial",18,"bold")).pack(pady=10)

resume_text = ctk.CTkTextbox(right_frame, width=400, height=500)
resume_text.pack(padx=10,pady=10)

app.mainloop()