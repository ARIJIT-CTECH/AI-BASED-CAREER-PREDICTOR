import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Training or loading model
def train_and_save_model():
    data = pd.read_csv('student_footprints.csv')

    data = data.join(data['Preferred_Languages'].str.get_dummies(sep=','))
    data.drop(['Preferred_Languages', 'Student_ID'], axis=1, inplace=True)

    X = data.drop('Career_Label', axis=1)
    y = data['Career_Label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with open('career_model.pkl', 'wb') as f:
        pickle.dump((model, scaler, list(X.columns)), f)

    print("✅ Model trained and saved!")

def load_model():
    with open('career_model.pkl', 'rb') as f:
        model, scaler, feature_names = pickle.load(f)
    return model, scaler, feature_names

# Predict for single student in CLI
def predict_single_cli(model, scaler, feature_names):
    print("\n--- Predict for One Student (CLI Mode) ---")

    user_input = {}

    for feature in feature_names:
        if feature in ['Python', 'C++', 'Java', 'HTML', 'CSS', 'JS', 'Go', 'Rust', 'SQL']:
            ans = input(f"Do you know {feature}? (yes/no): ").lower()
            user_input[feature] = 1 if ans == 'yes' else 0
        else:
            value = input(f"Enter your {feature.replace('_', ' ')}: ")
            try:
                user_input[feature] = float(value)
            except:
                print(f"Invalid input for {feature}. Setting 0.")
                user_input[feature] = 0.0

    input_features = [user_input[feat] for feat in feature_names]
    input_features = np.array(input_features).reshape(1, -1)
    input_scaled = scaler.transform(input_features)

    probabilities = model.predict_proba(input_scaled)[0]
    classes = model.classes_

    top3_indices = np.argsort(probabilities)[::-1][:3]
    top3_careers = [(classes[i], probabilities[i]*100) for i in top3_indices]

    print("\n🎯 Your Top 3 Career Recommendations:")

    for idx, (career, prob) in enumerate(top3_careers, start=1):
        print(f"{idx}. {career} ({prob:.2f}%)")

    # Bar Chart
    careers = [career for career, _ in top3_careers]
    probs = [prob for _, prob in top3_careers]

    plt.barh(careers[::-1], probs[::-1])
    plt.xlabel('Probability (%)')
    plt.title('Top 3 Career Recommendations')
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.show()

# Predict for multiple students (Bulk CSV)
def predict_bulk(model, scaler, feature_names):
    print("\n--- Predict for Multiple Students (CSV) ---")
    file_path = input("Enter path to your student CSV file: ")

    if not os.path.exists(file_path):
        print("❌ ERROR: File not found.")
        return

    students = pd.read_csv(file_path)

    # Handle missing columns
    for lang in ['Python', 'C++', 'Java', 'HTML', 'CSS', 'JS', 'Go', 'Rust', 'SQL']:
        students[lang] = students['Preferred_Languages'].apply(lambda x: 1 if lang in str(x).split(',') else 0)

    students = students.drop(['Preferred_Languages', 'Student_ID'], axis=1, errors='ignore')

    missing_cols = set(feature_names) - set(students.columns)
    for col in missing_cols:
        students[col] = 0

    students = students[feature_names]

    students_scaled = scaler.transform(students)

    predictions = model.predict(students_scaled)

    students['Predicted_Career_Path'] = predictions

    output_file = 'predicted_careers.csv'
    students.to_csv(output_file, index=False)

    print(f"🎯 Bulk predictions completed! Results saved to '{output_file}'.")

# GUI Prediction
def start_gui(model, scaler, feature_names):
    root = tk.Tk()
    root.title("Career Path AI Advisor")
    root.geometry("600x700")

    entries = {}
    language_vars = {}

    frame_inputs = tk.Frame(root)
    frame_inputs.pack(pady=10)

    fields = [feat for feat in feature_names if feat not in ['Python', 'C++', 'Java', 'HTML', 'CSS', 'JS', 'Go', 'Rust', 'SQL']]

    for field in fields:
        lbl = tk.Label(frame_inputs, text=field.replace('_', ' '))
        lbl.pack()
        entry = tk.Entry(frame_inputs)
        entry.pack()
        entries[field] = entry

    frame_lang = tk.LabelFrame(root, text="Select Known Languages")
    frame_lang.pack(pady=10)

    for lang in ['Python', 'C++', 'Java', 'HTML', 'CSS', 'JS', 'Go', 'Rust', 'SQL']:
        var = tk.IntVar()
        chk = tk.Checkbutton(frame_lang, text=lang, variable=var)
        chk.pack(anchor='w')
        language_vars[lang] = var

    def predict_career_gui():
        user_input = {}

        try:
            for field, entry in entries.items():
                user_input[field] = float(entry.get())

            for lang, var in language_vars.items():
                user_input[lang] = var.get()

            input_features = [user_input[feat] for feat in feature_names]
            input_features = np.array(input_features).reshape(1, -1)
            input_scaled = scaler.transform(input_features)

            probabilities = model.predict_proba(input_scaled)[0]
            classes = model.classes_

            top3_indices = np.argsort(probabilities)[::-1][:3]
            top3_careers = [(classes[i], probabilities[i]*100) for i in top3_indices]

            msg = "Top 3 Career Recommendations:\n\n"
            for idx, (career, prob) in enumerate(top3_careers, start=1):
                msg += f"{idx}. {career} ({prob:.2f}%)\n"

            messagebox.showinfo("Career Recommendations", msg)

            careers = [career for career, _ in top3_careers]
            probs = [prob for _, prob in top3_careers]

            plt.barh(careers[::-1], probs[::-1])
            plt.xlabel('Probability (%)')
            plt.title('Top 3 Career Recommendations')
            plt.xlim(0, 100)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    btn = tk.Button(root, text="Predict Career", command=predict_career_gui)
    btn.pack(pady=20)

    root.mainloop()

# Main program
def main():
    print("🚀 Welcome to Career Path AI!")

    if not os.path.exists('student_footprints.csv'):
        print("❌ ERROR: student_footprints.csv not found.")
        return

    if not os.path.exists('career_model.pkl'):
        print("⚙️ No model found, training model...")
        train_and_save_model()

    model, scaler, feature_names = load_model()

    while True:
        print("\n--- Main Menu ---")
        print("1. Predict Career for One Student (CLI)")
        print("2. Predict Career for Multiple Students (Bulk CSV)")
        print("3. Launch GUI Application")
        print("4. Exit")

        choice = input("Enter your choice (1/2/3/4): ")

        if choice == '1':
            predict_single_cli(model, scaler, feature_names)
        elif choice == '2':
            predict_bulk(model, scaler, feature_names)
        elif choice == '3':
            start_gui(model, scaler, feature_names)
        elif choice == '4':
            print("👋 Exiting. Goodbye!")
            break
        else:
            print("❌ Invalid choice. Try again.")

if __name__ == "__main__":
    main()
