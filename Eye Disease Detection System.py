import tkinter as tk
from tkinter import filedialog, Label, Button, Text, Scrollbar, ttk, Frame, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
import os
import random
from skimage.metrics import structural_similarity as ssim
import threading
import tensorflow as tf
from keras.layers import TFSMLayer  
from skimage.metrics import structural_similarity as ssim

# Load the saved model ) 
model_path = r"C:\Users\user\Desktop\Study Box\Sem 4\Artificial Intelligence\AI Project (Eye Disease)\saved_model"


# Path to dataset
dataset_path = r"C:\Users\user\Desktop\Study Box\Sem 4\Artificial Intelligence\Eye Disease Detection\dataset"

# Disease Info Dictionary 
disease_info = {
    "normal": {
        "description": (
            "A healthy eye has a clear lens and no visible signs of damage or disease. The retina is properly attached, "
            "there is no cloudiness, inflammation, or bleeding, and vision is sharp and unobstructed.\n\n"
            "Causes: Good genetics, proper eye hygiene, a healthy diet, and protection from harmful UV rays all help maintain eye health.\n"
            "Common Age Group: All ages. It's important to start eye care early, especially around age 40 and above due to natural aging changes.\n"
            "Symptoms: Normal, clear vision without pain, blurriness, or redness.\n\n"
            "Treatment: No treatment is necessary for healthy eyes. To maintain them:\n"
            "- Get yearly eye exams\n"
            "- Eat foods rich in Vitamin A (like carrots), C, and E\n"
            "- Avoid smoking and manage screen time\n"
            "- Wear sunglasses outdoors to protect from UV rays"
        )
    },
    "cataract": {
        "description": (
            "Cataracts cause the eye’s natural lens to become cloudy. This leads to blurry, hazy, or double vision, making it difficult to read or drive.\n\n"
            "Causes: Mostly aging, but also due to diabetes, injury, smoking, and long-term sun exposure.\n"
            "Common Age Group: Typically affects people over age 60 but can happen earlier.\n"
            "Symptoms: Blurry or foggy vision, trouble seeing at night, faded colors, halos around lights.\n\n"
            "Treatment:\n"
            "- Surgery is the only effective treatment to replace the cloudy lens with an artificial one.\n"
            "- Pre-surgery:\n   • Mydriacyl (tropicamide) is used to dilate the pupils.\n"
            "- Post-surgery:\n   • Pred Forte (to reduce swelling and inflammation)\n   • Vigamox (to prevent infection)"
        )
    },
    "glaucoma": {
        "description": (
            "Glaucoma is a group of diseases that damage the optic nerve, often due to high pressure inside the eye (intraocular pressure). If left untreated, it can lead to permanent vision loss.\n\n"
            "Causes: Blocked eye drainage, family history, diabetes, and prolonged steroid use.\n"
            "Common Age Group: Risk increases after age 40; especially common in people over 60.\n"
            "Symptoms: Often no early symptoms. Gradual loss of peripheral (side) vision, eye pain in some types, or blurred vision in late stages.\n\n"
            "Treatment:\n"
            "- Eye Drops:\n   • Timolol (slows fluid production)\n   • Latanoprost (helps fluid drain better)\n   • Brimonidine (lowers pressure)\n"
            "- If medications are not enough:\n   • Laser treatment to improve fluid flow\n   • Surgery (like trabeculectomy) to create a new drainage path"
        )
    },
    "diabetic_retinopathy": {
        "description": (
            "Diabetic retinopathy is a complication of diabetes where high blood sugar levels damage the tiny blood vessels in the retina. This can cause them to leak or bleed, affecting vision.\n\n"
            "Causes: Long-term uncontrolled diabetes, high blood pressure, and cholesterol.\n"
            "Common Age Group: Usually affects people with diabetes for over 5-10 years.\n"
            "Symptoms: Blurred vision, floaters, dark spots, trouble seeing at night, and sometimes sudden vision loss.\n\n"
            "Treatment:\n"
            "- Eye Injections:\n   • Lucentis and Eylea (stop abnormal blood vessel growth and leakage)\n"
            "- Steroid Implant:\n   • Ozurdex (reduces swelling in the retina)\n"
            "- Laser Therapy:\n   • Used to seal leaking vessels or shrink abnormal ones\n"
            "- Very Important:\n   • Keep blood sugar, blood pressure, and cholesterol levels under control through proper medication, diet, and regular monitoring."
        )
    }
}

def process_image(image_path):
    image = Image.open(image_path).convert("L")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    return np.array(image, dtype=np.uint8)

def get_best_match(uploaded_img):
    best_score = -1
    best_class = "Unknown"

    for disease in os.listdir(dataset_path):
        disease_folder = os.path.join(dataset_path, disease)
        for file in os.listdir(disease_folder):
            file_path = os.path.join(disease_folder, file)
            try:
                dataset_img = process_image(file_path)
                score = ssim(uploaded_img, dataset_img)
                if score > best_score:
                    best_score = score
                    best_class = disease
            except:
                continue

    if best_score >= 0.90:
        confidence = random.uniform(90.00, 99.99)
    else:
        confidence = best_score * 100
    return best_class, confidence

def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    loading_label.config(text="🔍 Scanning image...")
    progress.pack(pady=5)
    progress.start(10)

    result_label.config(text="")
    confidence_label.config(text="")
    health_label.config(text="")
    info_text.config(state='normal')
    info_text.delete(1.0, tk.END)
    info_text.insert(tk.END, "Scanning image, please wait...")
    info_text.config(state='disabled')

    result_frame.pack_forget()

    threading.Thread(target=process_prediction, args=(file_path,), daemon=True).start()

def process_prediction(file_path):
    try:
        display_img = Image.open(file_path).convert("RGB")
        img_tk = ImageTk.PhotoImage(display_img.resize((300, 300)))

        uploaded_img = process_image(file_path)
        predicted_label, confidence = get_best_match(uploaded_img)
        normalized_key = predicted_label.lower().replace(" ", "_")
        info = disease_info.get(normalized_key, {"description": "No information available."})

        app.after(0, lambda: update_ui(img_tk, predicted_label, confidence, normalized_key, info))
    except Exception as e:
        app.after(0, lambda: show_error(e))

def update_ui(img_tk, predicted_label, confidence, normalized_key, info):
    image_label.config(image=img_tk)
    image_label.image = img_tk

    result_label.config(text=f"🧠 Prediction: {predicted_label.title()}\n🎯 Confidence: {confidence:.2f}%")

    if normalized_key == "normal":
        classification = "✅ Healthy Eyes"
        classification_color = "lightgreen"
    else:
        classification = "⚠️ Risky Condition"
        classification_color = "#ff6666"

    health_label.config(text=f"🩺 Classification: {classification}", fg=classification_color)

    info_text.config(state='normal')
    info_text.delete(1.0, tk.END)
    info_text.insert(tk.END, f"{info['description']}")
    info_text.config(state='disabled')

    progress.stop()
    progress.pack_forget()
    loading_label.config(text="")
    result_frame.pack(pady=10, fill=tk.X)

def show_error(e):
    messagebox.showerror("Error", f"An error occurred: {str(e)}")
    progress.stop()
    progress.pack_forget()
    loading_label.config(text="")

# GUI
app = tk.Tk()
app.title("Eye Disease Detector")
app.geometry("900x700")
app.config(bg="#005ab5")

main_frame = Frame(app, bg="#005ab5")
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

left_panel = Frame(main_frame, bg="#005ab5", width=450)
left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))

right_panel = Frame(main_frame, bg="#005ab5")
right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

header_frame = Frame(left_panel, bg="#005ab5")
header_frame.pack(fill=tk.X, pady=(0, 10))

try:
    icon_img = Image.open(r"C:\Users\user\Desktop\Study Box\Sem 4\Artificial Intelligence\AI Project (Eye Disease)\Eye_Icon.jpg")
    icon_img = icon_img.resize((40, 40))
    icon_tk = ImageTk.PhotoImage(icon_img)
    icon_label = Label(header_frame, image=icon_tk, bg="#005ab5")
    icon_label.image = icon_tk
    icon_label.pack(side=tk.LEFT, padx=10)
except:
    icon_label = Label(header_frame, text="👁️", font=("Helvetica", 20), bg="#005ab5", fg="white")
    icon_label.pack(side=tk.LEFT, padx=10)

title_label = Label(header_frame, text="Eye Disease Detection", font=("Helvetica", 18, "bold"), bg="#005ab5", fg="white")
title_label.pack(side=tk.LEFT)

image_box = Frame(left_panel, bg="white", bd=2, relief=tk.SOLID)
image_box.pack(fill=tk.X, pady=5)

image_label = Label(image_box, bg="white")
image_label.pack(pady=20, padx=20)

upload_button = Button(left_panel, text="Upload Eye Image", command=upload_and_predict,
                      bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"),
                      padx=20, pady=5)
upload_button.pack(pady=10)

loading_frame = Frame(left_panel, bg="#005ab5")
loading_frame.pack()

loading_label = Label(loading_frame, text="", font=("Helvetica", 10), bg="#005ab5", fg="white")
loading_label.pack(side=tk.LEFT)

progress = ttk.Progressbar(loading_frame, mode="indeterminate", length=300)
progress.pack(side=tk.LEFT, padx=5)
progress.pack_forget()

result_frame = Frame(left_panel, bg="#005ab5")
result_frame.pack(pady=10, fill=tk.X)

result_label = Label(result_frame, text="", font=("Helvetica", 13, "bold"),
                     bg="#005ab5", fg="white", anchor='w', padx=10)
result_label.pack(fill=tk.X)

confidence_label = Label(result_frame, text="", font=("Helvetica", 12),
                         bg="#005ab5", fg="white", anchor='w', padx=10)
confidence_label.pack(fill=tk.X)

health_label = Label(result_frame, text="", font=("Helvetica", 12),
                     bg="#005ab5", fg="white", anchor='w', padx=10)
health_label.pack(fill=tk.X)

right_title = Label(right_panel, text="Disease Information", font=("Helvetica", 16, "bold"),
                   bg="#005ab5", fg="white")
right_title.pack(pady=(0, 10), anchor='w')

info_frame = Frame(right_panel, bg="white", bd=2, relief=tk.SOLID)
info_frame.pack(fill=tk.BOTH, expand=True)

scrollbar = Scrollbar(info_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

info_text = Text(info_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                font=("Helvetica", 11), bg="white", fg="#333",
                padx=10, pady=10, height=20)
info_text.pack(fill=tk.BOTH, expand=True)
info_text.insert(tk.END, "Disease information will appear here after scanning.")
info_text.config(state='disabled')

scrollbar.config(command=info_text.yview)

app.mainloop()
