import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template

# Load Data
movies  = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("https://drive.google.com/uc?id=1K1CsnFhukWffYh5TihYG5XKc2XvPaKAG")
movies = movies.merge(
    credits[["movie_id","cast","crew"]],
    left_on="id", right_on="movie_id", how="left"
)

# Preprocessing
def parse_names(field, limit=None):
    try:
        items = json.loads(field)
        names = [i["name"] for i in items if "name" in i]
        return names[:limit] if limit else names
    except:
        return []

def get_director(crew_str):
    try:
        for c in json.loads(crew_str):
            if c.get("job") == "Director":
                return c.get("name", "")
    except:
        pass
    return ""

movies["genres"]   = movies["genres"].apply(lambda x: parse_names(x))
movies["cast"]     = movies["cast"].apply(lambda x: parse_names(x, limit=5))
movies["keywords"] = movies["keywords"].apply(lambda x: parse_names(x, limit=10))
movies["director"] = movies["crew"].apply(get_director)
movies["year"]     = pd.to_datetime(
    movies["release_date"], errors="coerce"
).dt.year.fillna(0).astype(int)

movies = movies[
    (movies["vote_count"] >= 50) &
    (movies["vote_average"] > 0) &
    (movies["overview"].notna())
].reset_index(drop=True)

# Build Recommendation Engine
def build_soup(row):
    genres   = " ".join(row["genres"])
    cast     = " ".join([c.replace(" ","") for c in row["cast"]])
    director = row["director"].replace(" ","")
    keywords = " ".join([k.replace(" ","") for k in row["keywords"]])
    overview = str(row["overview"])
    return f"{genres} {genres} {cast} {director} {director} {keywords} {overview}"

movies["soup"] = movies.apply(build_soup, axis=1)

tfidf        = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies["soup"])
cosine_sim   = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices      = pd.Series(movies.index, index=movies["title"].str.lower())

def get_recommendations(title, n=12):
    title = title.lower()
    if title not in indices:
        return []
    idx = indices[title]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
    scores = sorted(
        list(enumerate(cosine_sim[idx])),
        key=lambda x: x[1], reverse=True
    )[1:n+1]
    movie_indices = [i[0] for i in scores]
    return movies.iloc[movie_indices][
        ["title","genres","vote_average","year","overview","director","cast"]
    ].to_dict("records")

# Flask App
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/movies")
def get_movies():
    data = []
    for _, row in movies.iterrows():
        data.append({
            "id": int(row["id"]),
            "t":  row["title"],
            "r":  round(float(row["vote_average"]), 1),
            "y":  int(row["year"]),
            "rt": int(row["runtime"]) if row["runtime"] else 0,
            "g":  row["genres"],
            "c":  row["cast"],
            "d":  row["director"],
            "o":  str(row["overview"])[:250],
        })
    return jsonify(data)

@app.route("/recommend")
def recommend():
    title = request.args.get("title", "")
    recs  = get_recommendations(title, n=12)
    for r in recs:
        match = movies[movies["title"] == r["title"]]
        r["id"]   = int(match["id"].iloc[0]) if len(match) else 0
        r["year"] = int(match["year"].iloc[0]) if len(match) else 0
    return jsonify(recs)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)