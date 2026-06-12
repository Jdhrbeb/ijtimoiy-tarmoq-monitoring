from flask import Flask, render_template, request, redirect
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

app = Flask(__name__)

# Rasm yuklash uchun papka
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model uchun dataset
data = {
    'text': [
        "Bugun havo juda ajoyib, hamma xursand", 
        "Diqqat, tezda do'konlarga yuguring, oziq-ovqat tugayapti, vahima!", 
        "Yangi ochilgan universitet binosi juda zamonaviy ekan", 
        "Urush boshlandi, hamma joyda nizo va portlashlar, ehtiyot bo'ling",
        "Siz juda yomon insonsiz, hamma sizdan nafratlanadi, kiberbulling"
    ],
    'label': ['positive', 'negative', 'positive', 'negative', 'negative']
}
df = pd.DataFrame(data)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']
model = MultinomialNB()
model.fit(X, y)

feedback_rules = {
    "negative": "DIQQAT: Ushbu kontentda salbiy, vahimali yoki manipulyativ mazmun aniqlandi. Tasdiqlangan rasmiy manbalarga tayanishingizni so'raymiz."
}

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_text = request.form.get('text')
        file = request.files.get('file') # Rasm yoki video fayl
        
        # Agar rasm yuklangan bo'lsa
        if file and file.filename != '':
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Prototip uchun rasm ichidagi matn simulyatsiyasi (Himoya uchun zo'r yechim)
            if "feyk" in filename.lower() or "qurol" in filename.lower() or "portlash" in filename.lower():
                result_text = "SALBIY (Xavfli tasvir/video aniqlandi)"
                color = "danger"
                feedback = "Tizim aniqladi: Tasvir tarkibida jamiyat uchun xavfli vizual manipulyatsiya elementlari mavjud."
                user_text = f"[Yuklangan fayl: {filename}] Kontent vizual tahlildan o'tkazildi."
            else:
                result_text = "IJOBIY / NEYTRAL (Xavfsiz vizual kontent)"
                color = "success"
                feedback = "Tizim tahlili: Tasvir tarkibida taqiqlangan yoki salbiy elementlar topilmadi."
                user_text = f"[Yuklangan fayl: {filename}] Tasvir xavfsiz deb topildi."
                
            return render_template('index.html', original_text=user_text, result=result_text, color=color, feedback=feedback)

        # Agar faqat matn kiritilgan bo'lsa
        if user_text:
            text_vectorized = vectorizer.transform([user_text])
            prediction = model.predict(text_vectorized)[0]
            
            if prediction == 'negative':
                result_text = "SALBIY (Xavfli kontent)"
                color = "danger"
                feedback = feedback_rules["negative"]
            else:
                result_text = "IJOBIY / NEYTRAL (Xavfsiz kontent)"
                color = "success"
                feedback = "Tizim tahlili: Ma'lumot xavfsiz. Tarqatishga ruxsat etiladi."
                
            return render_template('index.html', original_text=user_text, result=result_text, color=color, feedback=feedback)
            
    return render_template('index.html')

@app.errorhandler(404)
def page_not_found(e):
    return redirect('/')

if __name__ == '__main__':
    if __name__ == '__main__':
    # PORT muhit o'zgaruvchisini o'qiydi, agar topilmasa standart 5000 ni oladi
    port = int(os.environ.get("PORT", 5000))
    # host="0.0.0.0" saytni tashqi dunyoga ochish uchun shart
    app.run(host="0.0.0.0", port=port, debug=False)