# ---- Matplotlib headless backend FIX ----
import matplotlib
matplotlib.use('Agg')

# ---- Imports ----
from flask import Flask, render_template, request, send_file, redirect, Response
import sqlite3
import matplotlib.pyplot as plt
import io
from wordcloud import WordCloud
from model import predict_sentiment

# ---- Flask app ----
app = Flask(__name__)

# ---- Database setup ----
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS reviews(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            sentiment TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---- Main page ----
@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None

    if request.method == "POST":
        text = request.form["review"]
        sentiment = predict_sentiment(text)

        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("INSERT INTO reviews(text, sentiment) VALUES (?, ?)", (text, sentiment))
        conn.commit()
        conn.close()

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT text, sentiment FROM reviews ORDER BY id DESC LIMIT 10")
    reviews = c.fetchall()
    conn.close()

    return render_template("index.html", sentiment=sentiment, reviews=reviews)

# ---- Chart route ----
@app.route("/chart")
def chart():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT sentiment, COUNT(*) FROM reviews GROUP BY sentiment")
    data = c.fetchall()
    conn.close()

    if not data:
        data = [("No Data", 1)]

    labels = [d[0] for d in data]
    counts = [d[1] for d in data]

    fig, ax = plt.subplots(figsize=(4,4))
    ax.pie(counts, labels=labels, autopct='%1.1f%%')
    ax.set_title("Sentiment Distribution")

    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)

    return send_file(img, mimetype='image/png')

# ---- Word Cloud route ----
@app.route("/wordcloud")
def wordcloud():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT text FROM reviews")
    texts = c.fetchall()
    conn.close()

    all_text = " ".join([t[0] for t in texts])

    if not all_text.strip():
        all_text = "No Reviews Yet"

    wc = WordCloud(width=800, height=400, background_color='white').generate(all_text)

    img = io.BytesIO()
    wc.to_image().save(img, format='PNG')
    img.seek(0)

    return send_file(img, mimetype='image/png')

# ---- Clear history ----
@app.route("/clear", methods=["POST"])
def clear():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("DELETE FROM reviews")
    conn.commit()
    conn.close()
    return redirect("/")

# ---- Export CSV ----
@app.route("/export")
def export():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT text, sentiment FROM reviews")
    data = c.fetchall()
    conn.close()

    csv = "Review,Sentiment\n"
    for row in data:
        csv += f"{row[0]},{row[1]}\n"

    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=reviews.csv"}
    )

# ---- Run server ----
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


