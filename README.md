This program is an interactive ML-based movie recommender system.

It loads the TMDB dataset (or uses a small sample dataset if missing).

The data is preprocessed → cleaning text, extracting genres, handling ratings, popularity, and runtime.

It uses TF-IDF vectorization on movie overviews + genres.

Builds a cosine similarity matrix to find movies similar to a given title.

Applies K-Means clustering to group movies with similar features.

The system runs with an interactive menu offering options to:

Get movie recommendations by title.

Browse movies by genre.

Show popular movies.

Compare ML methods (cosine vs clustering).

Show detailed movie info.

Exit the system.

Main recommendation logic = cosine similarity (content-based).

Alternative logic = cluster-based recommendations.

👉 In short: it’s a content-based + clustering hybrid recommender, with an interactive user interface in the console.
