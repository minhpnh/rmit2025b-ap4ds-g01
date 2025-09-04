# from flask import Flask, render_template, request, redirect, url_for, session

# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Replace with a secure key

# @app.route('/')
# def index():
#     return render_template('home.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')


from flask import Flask, render_template, request, abort
import csv
import re

app = Flask(__name__)

# Load product data from CSV (no pandas)
def load_products():
    products = []
    with open("./data/webData.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Fill missing fields with empty string
            products.append({
                "id": row.get("id", "").strip(),
                "division": row.get("division", "").strip(),
                "department": row.get("department", "").strip(),
                "class": row.get("class", "").strip(),
                "title": row.get("title", "").strip(),
                "description": row.get("description", "").strip()
            })
    return products

products = load_products()
product_map = {p["id"]: p for p in products}

# === Tokenization and Search ===
def stem(word):
    word = word.lower()
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 3:
        return word[:-2]
    if word.endswith("s") and len(word) > 2:
        return word[:-1]
    return word

def tokenize(text):
    return {stem(w) for w in re.findall(r"\b\w+\b", text.lower())}

@app.template_filter("hl")
def highlight(text, q):
    if not q:
        return text
    tokens = {stem(w) for w in re.findall(r"\b\w+\b", q.lower())}
    return re.sub(r"\b\w+\b", lambda m: f"<mark>{m.group(0)}</mark>" if stem(m.group(0)) in tokens else m.group(0), text)

# === Routes ===
@app.route("/")
def index():
    return render_template("index.html", products=products[:24])

@app.route("/search")
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return render_template("index.html", products=products[:24], q=q, total=len(products))

    q_tokens = tokenize(q)

    def is_match(p):
        text = f"{p['title']} {p['description']} {p['division']} {p['department']} {p['class']}"
        return any(token in tokenize(text) for token in q_tokens)

    results = [p for p in products if is_match(p)]
    return render_template("index.html", products=results, q=q, total=len(results))

@app.route("/item/<id>")
def item(id):
    p = product_map.get(str(id))
    if not p:
        abort(404)
    return render_template("detail.html", p=p)

if __name__ == "__main__":
    app.run(debug=True)
