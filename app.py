from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from search_engine import (
    load_stop_words, preprocess, load_documents,
    compute_tf, compute_tfidf, cosine_similarity, search_images
)
from nltk.stem.snowball import SnowballStemmer
from pathlib import Path

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")


def run_query_search(query):
    stop_words = load_stop_words()
    stemmer = SnowballStemmer('english')
    processed = preprocess(query, True, True, stop_words, stemmer)

    img_titles, captions = load_documents('captions.txt', True, True, stop_words, stemmer)
    all_docs = captions + [processed]
    vocab, df = compute_tf(all_docs)
    vecs = compute_tfidf(all_docs, vocab, df, len(all_docs))

    query_vec = vecs[-1]
    caption_vecs = vecs[:-1]

    scores = [cosine_similarity(query_vec, cap_vec) for cap_vec in caption_vecs]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:15]
    for i in top_indices:
        img_titles[i] = Path(img_titles[i]).name
        print(img_titles[i])

    return [img_titles[i] for i in top_indices]


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "")
    top_images = run_query_search(query)
    return jsonify({"results": top_images})

@app.route("/search", methods=["GET"])
def search_get():
    query = request.args.get("query", "")
    top_images = run_query_search(query) if query else []
    return render_template("search_results.html", query=query, results=top_images)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/my_work')
def my_work():
    return render_template('my_work.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/current_proj_2')
def current_proj_2():
    return render_template('current_proj_2.html')

@app.route('/current_proj_3')
def current_proj_3():
    return render_template('current_proj_3.html')

@app.route('/current_proj')
def current_proj():
    return render_template('current_proj.html')

@app.route('/sports')
def sports():
    return render_template('sports.html')

@app.route('/grad_indv')
def grad_indv():
    return render_template('grad_indv.html')

@app.route('/land_travel')
def land_travel():
    return render_template('land_travel.html')


if __name__ == "__main__":
    app.run(debug=True)
