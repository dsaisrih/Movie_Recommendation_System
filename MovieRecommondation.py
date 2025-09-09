import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
import os

class InteractiveMLMovieRecommender:
    """
    Interactive Machine Learning Movie Recommendation System
    Features user input for real-time recommendations
    """
    
    def __init__(self):
        self.movies_df = None
        self.feature_vectors = {}
        self.similarity_matrices = {}
        self.vectorizers = {}
        self.scaler = StandardScaler()
        self.svd = TruncatedSVD(n_components=50, random_state=42)
        self.kmeans = KMeans(n_clusters=15, random_state=42)
        self.is_trained = False
        
    def setup_system(self):
        """
        Setup the ML recommendation system
        """
        print("🎬 INTERACTIVE ML MOVIE RECOMMENDATION SYSTEM")
        print("=" * 60)
        
        # Check if dataset exists
        movies_file = 'tmdb_5000_movies.csv'
        credits_file = 'tmdb_5000_credits.csv'
        
        if os.path.exists(movies_file):
            print("📁 Found dataset files!")
            self._load_real_data(movies_file, credits_file)
        else:
            print("📁 Dataset not found. Using sample data for demonstration.")
            self._create_sample_data()
            
        self._train_ml_models()
        self.is_trained = True
        print("✅ ML System Ready for Recommendations!")
        
    def _load_real_data(self, movies_file, credits_file):
        """Load real TMDB dataset"""
        print("🔄 Loading TMDB dataset...")
        
        self.movies_df = pd.read_csv(movies_file)
        
        if os.path.exists(credits_file):
            credits_df = pd.read_csv(credits_file)
            self.movies_df = self.movies_df.merge(credits_df, on='id', how='left')
            
        self._preprocess_real_data()
        
    def _create_sample_data(self):
        """Create sample data for demonstration"""
        sample_movies = {
            'id': range(1, 21),
            'title': [
                'The Dark Knight', 'Batman Begins', 'The Dark Knight Rises',
                'Inception', 'Interstellar', 'Dunkirk',
                'The Matrix', 'The Matrix Reloaded', 'The Matrix Revolutions',
                'Blade Runner', 'Blade Runner 2049', 'Alien',
                'Toy Story', 'Toy Story 2', 'Toy Story 3',
                'The Avengers', 'Iron Man', 'Captain America',
                'Star Wars', 'The Empire Strikes Back'
            ],
            'overview': [
                'Batman faces the Joker in this dark superhero thriller',
                'Bruce Wayne becomes Batman to fight crime in Gotham',
                'Batman faces Bane in the final chapter',
                'Dreams within dreams in this mind-bending thriller',
                'Space exploration and time dilation science fiction',
                'World War II Dunkirk evacuation war drama',
                'Reality is questioned in this cyberpunk action film',
                'Neo continues his journey in the Matrix',
                'The final battle for humanity in the Matrix',
                'Future detective hunts replicants in dystopian LA',
                'Sequel to Blade Runner set decades later',
                'Space horror with deadly alien creature',
                'Toys come to life in this animated adventure',
                'Toy Story continues with new adventures',
                'Andy grows up and toys find new home',
                'Superheroes assemble to save the world',
                'Billionaire becomes armored superhero',
                'Super soldier fights for America',
                'Space opera about rebellion against empire',
                'Empire strikes back against rebellion'
            ],
            'genres': [
                'Action Crime Drama', 'Action Crime Drama', 'Action Crime Drama',
                'Action Sci-Fi Thriller', 'Adventure Drama Sci-Fi', 'Action Drama War',
                'Action Sci-Fi', 'Action Sci-Fi', 'Action Sci-Fi',
                'Sci-Fi Thriller', 'Sci-Fi Thriller', 'Horror Sci-Fi',
                'Animation Comedy Family', 'Animation Comedy Family', 'Animation Comedy Family',
                'Action Adventure Sci-Fi', 'Action Adventure Sci-Fi', 'Action Adventure Sci-Fi',
                'Adventure Fantasy Sci-Fi', 'Adventure Fantasy Sci-Fi'
            ],
            'vote_average': [9.0, 8.2, 8.4, 8.8, 8.6, 7.9, 8.7, 7.2, 6.8, 8.1, 8.0, 8.4, 8.3, 7.9, 8.2, 8.0, 7.9, 6.9, 8.6, 8.7],
            'popularity': [100, 95, 90, 98, 85, 75, 96, 80, 70, 88, 82, 86, 92, 88, 89, 94, 91, 78, 99, 97],
            'runtime': [152, 140, 165, 148, 169, 106, 136, 138, 129, 117, 164, 117, 81, 99, 103, 143, 126, 124, 121, 124]
        }
        
        self.movies_df = pd.DataFrame(sample_movies)
        print(f"📊 Created sample dataset with {len(self.movies_df)} movies")
        
    def _preprocess_real_data(self):
        """Preprocess real TMDB data"""
        print("🔄 Preprocessing real data...")
        
        # Handle missing values
        self.movies_df['overview'] = self.movies_df['overview'].fillna('')
        self.movies_df['genres'] = self.movies_df['genres'].fillna('[]')
        
        # Extract genre names from JSON
        self.movies_df['genre_names'] = self.movies_df['genres'].apply(self._extract_genre_names)
        
        # Create combined text features
        self.movies_df['combined_features'] = (
            self.movies_df['overview'] + ' ' + 
            self.movies_df['genre_names']
        ).apply(self._clean_text)
        
        # Handle numerical features
        self.movies_df['popularity'] = pd.to_numeric(self.movies_df.get('popularity', 0), errors='coerce').fillna(0)
        self.movies_df['vote_average'] = pd.to_numeric(self.movies_df.get('vote_average', 0), errors='coerce').fillna(0)
        self.movies_df['runtime'] = pd.to_numeric(self.movies_df.get('runtime', 0), errors='coerce').fillna(0)
        
    def _extract_genre_names(self, genres_str):
        """Extract genre names from JSON string"""
        import ast
        try:
            genres = ast.literal_eval(genres_str)
            return ' '.join([genre['name'].replace(' ', '') for genre in genres])
        except:
            return ''
            
    def _clean_text(self, text):
        """Clean text for ML processing"""
        import re
        if pd.isna(text):
            return ''
        text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower())
        return re.sub(r'\s+', ' ', text).strip()
        
    def _train_ml_models(self):
        """Train ML models"""
        print("🤖 Training ML Models...")
        
        # For sample data, use simpler feature extraction
        if 'combined_features' not in self.movies_df.columns:
            self.movies_df['combined_features'] = (
                self.movies_df['overview'] + ' ' + 
                self.movies_df['genres']
            ).apply(self._clean_text)
        
        # TF-IDF Vectorization
        self.vectorizers['tfidf'] = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        tfidf_matrix = self.vectorizers['tfidf'].fit_transform(
            self.movies_df['combined_features']
        )
        
        # Store feature vectors
        self.feature_vectors['tfidf'] = tfidf_matrix
        
        # Calculate similarity matrices
        self.similarity_matrices['cosine'] = cosine_similarity(tfidf_matrix)
        
        # K-Means Clustering
        if tfidf_matrix.shape[0] > 3:  # Only if we have enough movies
            n_clusters = min(5, len(self.movies_df) // 2)
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.movies_df['cluster'] = self.kmeans.fit_predict(tfidf_matrix.toarray())
        else:
            self.movies_df['cluster'] = 0
            
        print("✅ ML Models Trained!")
        
    def run_interactive_system(self):
        """
        Main interactive loop for user input
        """
        if not self.is_trained:
            print("❌ System not trained yet! Please run setup_system() first.")
            return
            
        print("\n🎯 INTERACTIVE MOVIE RECOMMENDATION SYSTEM")
        print("=" * 60)
        
        while True:
            self._display_menu()
            choice = input("\n👉 Enter your choice (1-6): ").strip()
            
            if choice == '1':
                self._get_recommendations_by_title()
            elif choice == '2':
                self._browse_by_genre()
            elif choice == '3':
                self._get_popular_movies()
            elif choice == '4':
                self._compare_ml_methods()
            elif choice == '5':
                self._show_movie_details()
            elif choice == '6':
                print("\n👋 Thank you for using ML Movie Recommender!")
                break
            else:
                print("❌ Invalid choice! Please try again.")
                
            input("\n📱 Press Enter to continue...")
            
    def _display_menu(self):
        """Display interactive menu"""
        print("\n" + "="*60)
        print("🎬 WHAT WOULD YOU LIKE TO DO?")
        print("="*60)
        print("1️⃣  Get Movie Recommendations (by title)")
        print("2️⃣  Browse Movies by Genre")
        print("3️⃣  Show Popular Movies")
        print("4️⃣  Compare ML Methods")
        print("5️⃣  Get Movie Details")
        print("6️⃣  Exit System")
        print("="*60)
        
    def _get_recommendations_by_title(self):
        """Interactive movie recommendations"""
        print("\n🎬 MOVIE RECOMMENDATION ENGINE")
        print("-" * 40)
        
        # Show available movies
        print("📚 Available movies (showing first 10):")
        for i, title in enumerate(self.movies_df['title'].head(10), 1):
            print(f"   {i}. {title}")
            
        if len(self.movies_df) > 10:
            print(f"   ... and {len(self.movies_df) - 10} more movies")
        
        movie_title = input("\n🔍 Enter movie title (or part of it): ").strip()
        
        if not movie_title:
            print("❌ Please enter a movie title!")
            return
            
        num_recommendations = self._get_number_input(
            "📊 How many recommendations do you want? (1-10): ", 1, 10, 5
        )
        
        recommendations = self._find_similar_movies(movie_title, num_recommendations)
        
        if recommendations:
            self._display_recommendations(recommendations, movie_title)
        else:
            self._suggest_similar_titles(movie_title)
            
    def _find_similar_movies(self, movie_title, num_recommendations):
        """Find similar movies using ML"""
        # Find matching movies
        movie_matches = self.movies_df[
            self.movies_df['title'].str.contains(movie_title, case=False, na=False)
        ]
        
        if movie_matches.empty:
            return []
            
        movie_idx = movie_matches.index[0]
        movie_info = movie_matches.iloc[0]
        
        # Get similarity scores
        similarity_scores = self.similarity_matrices['cosine'][movie_idx]
        
        # Get top similar movies
        top_indices = np.argsort(similarity_scores)[::-1][1:num_recommendations+1]
        
        recommendations = []
        for idx in top_indices:
            if idx < len(self.movies_df):
                movie = self.movies_df.iloc[idx]
                recommendations.append({
                    'title': movie['title'],
                    'rating': round(movie.get('vote_average', 0), 1),
                    'genres': movie.get('genres', 'N/A'),
                    'similarity_score': round(similarity_scores[idx], 3),
                    'cluster': movie.get('cluster', 0),
                    'overview': movie.get('overview', 'No overview available')[:100] + '...'
                })
                
        return recommendations
        
    def _display_recommendations(self, recommendations, input_title):
        """Display recommendations in a formatted way"""
        print(f"\n🎯 TOP RECOMMENDATIONS FOR: '{input_title}'")
        print("=" * 70)
        
        for i, movie in enumerate(recommendations, 1):
            print(f"\n{i}. 🎬 {movie['title']}")
            print(f"   ⭐ Rating: {movie['rating']}/10")
            print(f"   🎭 Genres: {movie['genres']}")
            print(f"   🤖 ML Similarity: {movie['similarity_score']}")
            print(f"   🏷️  Cluster: {movie['cluster']}")
            print(f"   📝 Overview: {movie['overview']}")
            print("-" * 50)
            
    def _browse_by_genre(self):
        """Browse movies by genre"""
        print("\n🎭 BROWSE BY GENRE")
        print("-" * 30)
        
        # Get unique genres
        all_genres = set()
        for genres_str in self.movies_df['genres']:
            if isinstance(genres_str, str):
                genres = genres_str.replace('[', '').replace(']', '').replace("'", '').split()
                all_genres.update(genres)
        
        genre_list = sorted(list(all_genres))[:10]  # Show first 10 genres
        
        print("🎨 Available genres:")
        for i, genre in enumerate(genre_list, 1):
            print(f"   {i}. {genre}")
            
        genre_input = input("\n🔍 Enter genre name: ").strip()
        
        if not genre_input:
            print("❌ Please enter a genre!")
            return
            
        # Find movies with matching genre
        genre_movies = self.movies_df[
            self.movies_df['genres'].str.contains(genre_input, case=False, na=False)
        ].sort_values('vote_average', ascending=False).head(10)
        
        if genre_movies.empty:
            print(f"❌ No movies found for genre: {genre_input}")
            return
            
        print(f"\n🎬 TOP {genre_input.upper()} MOVIES:")
        print("=" * 50)
        
        for i, (_, movie) in enumerate(genre_movies.iterrows(), 1):
            print(f"{i}. {movie['title']} - ⭐ {movie.get('vote_average', 0)}/10")
            
    def _get_popular_movies(self):
        """Show popular movies"""
        print("\n🔥 POPULAR MOVIES")
        print("-" * 25)
        
        num_movies = self._get_number_input(
            "📊 How many popular movies to show? (1-20): ", 1, 20, 10
        )
        
        popular_movies = self.movies_df.nlargest(num_movies, 'vote_average')
        
        print(f"\n🏆 TOP {num_movies} POPULAR MOVIES:")
        print("=" * 40)
        
        for i, (_, movie) in enumerate(popular_movies.iterrows(), 1):
            print(f"{i:2d}. {movie['title']}")
            print(f"     ⭐ {movie.get('vote_average', 0)}/10 | 🔥 Popularity: {movie.get('popularity', 0):.0f}")
            print()
            
    def _compare_ml_methods(self):
        """Compare different ML approaches"""
        print("\n🔬 ML METHODS COMPARISON")
        print("-" * 35)
        
        movie_title = input("🔍 Enter movie title to compare methods: ").strip()
        
        if not movie_title:
            print("❌ Please enter a movie title!")
            return
            
        print(f"\n📊 COMPARING ML METHODS FOR: '{movie_title}'")
        print("=" * 60)
        
        # Method 1: Cosine Similarity
        recommendations_cosine = self._find_similar_movies(movie_title, 3)
        
        print("\n🤖 METHOD 1: Cosine Similarity (TF-IDF)")
        print("-" * 40)
        if recommendations_cosine:
            for i, movie in enumerate(recommendations_cosine, 1):
                print(f"   {i}. {movie['title']} (Score: {movie['similarity_score']})")
        else:
            print("   No recommendations found")
            
        # Method 2: Cluster-based
        cluster_recs = self._get_cluster_recommendations(movie_title, 3)
        
        print("\n🎯 METHOD 2: K-Means Clustering")
        print("-" * 40)
        if cluster_recs:
            for i, movie in enumerate(cluster_recs, 1):
                print(f"   {i}. {movie['title']} (Cluster: {movie['cluster']})")
        else:
            print("   No recommendations found")
            
    def _get_cluster_recommendations(self, movie_title, num_recs):
        """Get recommendations based on clustering"""
        movie_matches = self.movies_df[
            self.movies_df['title'].str.contains(movie_title, case=False, na=False)
        ]
        
        if movie_matches.empty:
            return []
            
        movie_cluster = movie_matches.iloc[0]['cluster']
        
        # Get movies from same cluster
        cluster_movies = self.movies_df[
            (self.movies_df['cluster'] == movie_cluster) & 
            (~self.movies_df['title'].str.contains(movie_title, case=False, na=False))
        ].sort_values('vote_average', ascending=False).head(num_recs)
        
        recommendations = []
        for _, movie in cluster_movies.iterrows():
            recommendations.append({
                'title': movie['title'],
                'cluster': movie['cluster'],
                'rating': movie.get('vote_average', 0)
            })
            
        return recommendations
        
    def _show_movie_details(self):
        """Show detailed movie information"""
        print("\n📋 MOVIE DETAILS")
        print("-" * 20)
        
        movie_title = input("🔍 Enter movie title: ").strip()
        
        movie_info = self.movies_df[
            self.movies_df['title'].str.contains(movie_title, case=False, na=False)
        ]
        
        if movie_info.empty:
            print(f"❌ Movie '{movie_title}' not found!")
            self._suggest_similar_titles(movie_title)
            return
            
        movie = movie_info.iloc[0]
        
        print(f"\n🎬 MOVIE DETAILS: {movie['title']}")
        print("=" * 50)
        print(f"⭐ Rating: {movie.get('vote_average', 'N/A')}/10")
        print(f"🔥 Popularity: {movie.get('popularity', 'N/A')}")
        print(f"⏱️  Runtime: {movie.get('runtime', 'N/A')} minutes")
        print(f"🎭 Genres: {movie.get('genres', 'N/A')}")
        print(f"🏷️  ML Cluster: {movie.get('cluster', 'N/A')}")
        print(f"\n📝 Overview:")
        print(f"   {movie.get('overview', 'No overview available')}")
        
    def _suggest_similar_titles(self, movie_title):
        """Suggest similar movie titles"""
        similar_titles = self.movies_df[
            self.movies_df['title'].str.contains(
                movie_title.split()[0], case=False, na=False
            )
        ]['title'].head(5)
        
        if not similar_titles.empty:
            print(f"\n🔍 Did you mean one of these?")
            for i, title in enumerate(similar_titles, 1):
                print(f"   {i}. {title}")
                
    def _get_number_input(self, prompt, min_val, max_val, default):
        """Get number input with validation"""
        try:
            user_input = input(prompt).strip()
            if not user_input:
                return default
            num = int(user_input)
            return max(min_val, min(max_val, num))
        except ValueError:
            print(f"⚠️  Invalid number. Using default: {default}")
            return default

def main():
    """
    Main function to run the interactive system
    """
    recommender = InteractiveMLMovieRecommender()
    
    try:
        # Setup the system
        recommender.setup_system()
        
        # Run interactive loop
        recommender.run_interactive_system()
        
    except KeyboardInterrupt:
        print("\n\n👋 Thank you for using ML Movie Recommender!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your dataset files or try with sample data.")

if __name__ == "__main__":
    main()