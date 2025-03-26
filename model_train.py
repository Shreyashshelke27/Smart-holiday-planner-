import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Load the datasets
users = pd.read_csv('datasets/Users.csv')
destinations = pd.read_csv('datasets/Destinations.csv')
user_history = pd.read_csv('datasets/Userhistory.csv')
reviews = pd.read_csv('datasets/Review.csv')

# Merge datasets
merged_data = pd.merge(user_history, reviews, on=['DestinationID', 'UserID'], how='left')
merged_data = pd.merge(merged_data, users, on='UserID', how='left')
merged_data = pd.merge(merged_data, destinations, on='DestinationID', how='left')


# Content-Based Filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(merged_data['Category'] + " " + merged_data['TravelPreferences'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_content_based_recommendations(user_id, top_n=10):
    indices = merged_data[merged_data['UserID'] == user_id].index
    if len(indices) == 0:
        return np.array([])
    sim_scores = cosine_sim[indices].mean(axis=0)
    return merged_data.iloc[sim_scores.argsort()[-(top_n+1):-1][::-1]]['DestinationID'].unique()


# Collaborative Filtering (User-Item Matrix)
user_item_matrix = merged_data.pivot_table(index='UserID', columns='DestinationID', values='Rating', fill_value=0)
user_item_matrix_sparse = csr_matrix(user_item_matrix.values)
user_similarity = cosine_similarity(user_item_matrix_sparse)

def get_collaborative_based_recommendations(user_id, top_n=10):
    try:
        user_idx = user_item_matrix.index.get_loc(user_id)
    except KeyError:
        return np.array([])
    
    sim_scores = user_similarity[user_idx]
    similar_users = sim_scores.argsort()[::-1][1:top_n+1]
    
    return user_item_matrix.iloc[similar_users].mean(axis=0).sort_values(ascending=False).index[:top_n]

# Hybrid Model
def get_hybrid_recommendations(user_id, top_n=10):
    content_based_recs = get_content_based_recommendations(user_id, top_n)
    collaborative_based_recs = get_collaborative_based_recommendations(user_id, top_n)
    
    hybrid_recs = list(set(content_based_recs).union(set(collaborative_based_recs)))
    return hybrid_recs[:top_n]


# Evaluate the model

def evaluate_hybrid_model(top_n=10):
    metrics = {'precision': [], 'recall': [], 'f1': [], 'rmse': []}

    for user_id in users['UserID'].unique():
        try:
            actual = merged_data.loc[merged_data['UserID'] == user_id, 'DestinationID'].values
            predicted = get_hybrid_recommendations(user_id, top_n)
            
            if not predicted:
                continue  # Skip if no recommendations

            # Convert to binary format
            actual_binary = np.isin(user_item_matrix.columns, actual)
            predicted_binary = np.isin(user_item_matrix.columns, predicted)

            metrics['precision'].append(precision_score(actual_binary, predicted_binary, average='micro'))
            metrics['recall'].append(recall_score(actual_binary, predicted_binary, average='micro'))
            metrics['f1'].append(f1_score(actual_binary, predicted_binary, average='micro'))
            metrics['rmse'].append(np.sqrt(mean_squared_error(actual_binary, predicted_binary)))
        
        except Exception as e:
            print(f"Error evaluating User {user_id}: {e}")

    # Print averaged evaluation metrics
    print("\nHybrid Model Evaluation Metrics:")
    for metric, values in metrics.items():
        print(f"\n{metric.capitalize()}: {np.mean(values):.2f}")

# Run evaluation
evaluate_hybrid_model()


import pickle

# Create an instance of the model
hybrid_model = {
    'tfidf_vectorizer': tfidf,
    'tfidf_matrix': tfidf_matrix,
    'cosine_sim_matrix': cosine_sim,
    'user_item_matrix': user_item_matrix,
    'user_similarity_matrix': user_similarity,
    'merged_data': merged_data,
    'users': users,
    'destinations': destinations
}

# Save the model to a file
with open('models/hybrid_recommendation_model.pkl', 'wb') as f:
    pickle.dump(hybrid_model, f)

print("\n✅ Hybrid recommendation model has been saved successfully\n")
