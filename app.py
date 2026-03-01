from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 🔥 Load dataset properly
movies = pd.read_csv(
    'movies.csv',
    quotechar='"',
    skipinitialspace=True,
    dtype=str   # Force all columns as string
)

# 🔥 Clean whitespace from all string columns
movies = movies.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Fill missing values just in case
movies.fillna("", inplace=True)

# Create combined features for similarity
movies['combined'] = movies['genre'] + " " + movies['overview']

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['combined'])
similarity = cosine_similarity(tfidf_matrix)


def recommend(movie_name, genre_filter=None):
    if movie_name not in movies['title'].values:
        return []

    idx = movies[movies['title'] == movie_name].index[0]
    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommended = []

    for i in sorted_scores:
        movie = movies.iloc[i[0]]

        if movie['title'] == movie_name:
            continue

        if genre_filter and movie['genre'] != genre_filter:
            continue

        recommended.append(movie)

    return recommended[:8]


@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []

    if request.method == 'POST':
        selected_movie = request.form.get('movie')
        genre_filter = request.form.get('genre')
        recommendations = recommend(selected_movie, genre_filter)

    genres = sorted(movies['genre'].unique())

    return render_template(
        'index.html',
        movies=movies['title'].values,
        genres=genres,
        recommendations=recommendations
    )


# ✅ MOVIE DETAILS ROUTE WITH CLEAN CAST + SIMILAR MOVIES
@app.route('/movie/<title>')
def movie_details_page(title):

    if title not in movies['title'].values:
        return "Movie not found", 404

    idx = movies[movies['title'] == title].index[0]
    movie = movies.iloc[idx]

    # 🔥 Properly split cast into list
    cast_list = []
    if movie['cast']:
        cast_list = [actor.strip() for actor in movie['cast'].split(",")]

    # Similar movies logic
    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    similar_movies = []
    for i in sorted_scores[1:5]:
        similar_movies.append(movies.iloc[i[0]])

    return render_template(
        'movie_details.html',
        movie=movie,
        cast_list=cast_list,
        similar_movies=similar_movies
    )


if __name__ == '__main__':
    app.run(debug=True)