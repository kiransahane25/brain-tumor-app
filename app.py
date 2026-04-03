import streamlit as st
import sqlite3
import cv2
import numpy as np
import json
import gdown
import os

# ---------------- LOAD MODEL (KERAS FIX) ----------------
from keras.models import load_model

# ---------------- DOWNLOAD MODELS ----------------
if not os.path.exists("model.h5"):
    gdown.download("https://drive.google.com/uc?id=1b74HEfoR66sNZfkqnVZELV6-zT4LtAvy", "model.h5", quiet=False)

if not os.path.exists("mri_model.h5"):
    gdown.download("https://drive.google.com/uc?id=10rZ1Kdsdwb-QhwTvgBuJf3yOV8ChVfCO", "mri_model.h5", quiet=False)

# ---------------- LOAD MODELS ----------------
mri_model = load_model("mri_model.h5")
tumor_model = load_model("model.h5")

# ---------------- LOAD CLASS MAPPING ----------------
with open("mri_class_indices.json", "r") as f:
    class_indices = json.load(f)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("users.db")
c = conn.cursor()

c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")
c.execute("CREATE TABLE IF NOT EXISTS history (username TEXT, result TEXT)")
conn.commit()

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# ---------------- IMAGE ENHANCEMENT ----------------
def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    contrast = cv2.equalizeHist(blur)

    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])

    sharpened = cv2.filter2D(contrast, -1, kernel)
    return sharpened

# ---------------- SIDEBAR ----------------
menu = ["Login", "Signup"]
choice = st.sidebar.selectbox("Menu", menu)

# ---------------- SIGNUP ----------------
if choice == "Signup":
    st.subheader("Create Account")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type='password')

    if st.button("Signup"):
        c.execute("INSERT INTO users VALUES (?,?)", (user, pwd))
        conn.commit()
        st.success("Account created!")

# ---------------- LOGIN ----------------
elif choice == "Login":
    st.subheader("Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type='password')

    if st.button("Login"):
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (user, pwd))
        data = c.fetchall()

        if data:
            st.session_state.logged_in = True
            st.session_state.username = user
            st.success("Login Successful!")
        else:
            st.error("Invalid Credentials")

# ---------------- MAIN APP ----------------
if st.session_state.logged_in:
    st.title("🧠 Brain Tumor Detection System")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.stop()

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, caption="Uploaded Image", width="stretch")

        # ---------------- MRI CHECK ----------------
        img = cv2.resize(image, (128,128)) / 255.0
        img = np.reshape(img, (1,128,128,3))

        mri_pred = mri_model.predict(img)
        value = float(mri_pred[0][0])

        st.write("MRI Score:", value)

        if value < 0.5:
            st.success("✅ Valid MRI Image")

            # ---------------- ENHANCEMENT ----------------
            enhanced = enhance_image(image)
            st.image(enhanced, caption="Enhanced Image", width="stretch")

            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

            img2 = cv2.resize(enhanced, (128,128)) / 255.0
            img2 = np.reshape(img2, (1,128,128,3))

            # ---------------- TUMOR PREDICTION ----------------
            prediction = tumor_model.predict(img2)

            if prediction[0][0] > 0.5:
                result = "Tumor Detected ❌"
                st.error(result)
            else:
                result = "No Tumor Detected ✅"
                st.success(result)

            st.write("Confidence:", float(prediction[0][0]))

            c.execute("INSERT INTO history VALUES (?,?)", (st.session_state.username, result))
            conn.commit()

        else:
            st.error("❌ Not a Brain MRI Image")

    # ---------------- HISTORY ----------------
    st.subheader("📜 Prediction History")

    c.execute("SELECT * FROM history WHERE username=?", (st.session_state.username,))
    data = c.fetchall()

    if data:
        for row in data:
            st.write(row[1])
    else:
        st.write("No history yet.")
