import streamlit as st
import sqlite3
import cv2
import numpy as np
import gdown
import os
import tflite_runtime.interpreter as tflite

# ---------------- DOWNLOAD MODELS ----------------
if not os.path.exists("mri_model.tflite"):
    gdown.download("https://drive.google.com/uc?id=10-mxEApJGteD51YhDFc_lZ-kqIPfG5L9", "mri_model.tflite", quiet=False)

if not os.path.exists("model.tflite"):
    gdown.download("https://drive.google.com/uc?id=1MHh8tFVIl0llglydt5q32X8UmHIrgKTX", "model.tflite", quiet=False)

# ---------------- LOAD MODELS ----------------
def load_model(path):
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

mri_interpreter = load_model("mri_model.tflite")
tumor_interpreter = load_model("model.tflite")

# ---------------- PREDICT FUNCTION ----------------
def predict(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image.astype(np.float32))
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return output

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
    return cv2.cvtColor(contrast, cv2.COLOR_GRAY2RGB)

# ---------------- UI ----------------
menu = ["Login", "Signup"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Signup":
    st.subheader("Create Account")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type='password')

    if st.button("Signup"):
        c.execute("INSERT INTO users VALUES (?,?)", (user, pwd))
        conn.commit()
        st.success("Account created!")

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

# ---------------- MAIN ----------------
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

        # MRI CHECK
        img = cv2.resize(image, (128,128)) / 255.0
        img = np.reshape(img, (1,128,128,3))

        mri_pred = predict(mri_interpreter, img)
        value = float(mri_pred[0][0])

        if value < 0.5:
            st.success("✅ Valid MRI Image")

            enhanced = enhance_image(image)
            st.image(enhanced, caption="Enhanced Image", width="stretch")

            img2 = cv2.resize(enhanced, (128,128)) / 255.0
            img2 = np.reshape(img2, (1,128,128,3))

            pred = predict(tumor_interpreter, img2)

            if pred[0][0] > 0.5:
                result = "Tumor Detected ❌"
                st.error(result)
            else:
                result = "No Tumor Detected ✅"
                st.success(result)

            c.execute("INSERT INTO history VALUES (?,?)", (st.session_state.username, result))
            conn.commit()

        else:
            st.error("❌ Not a Brain MRI Image")
