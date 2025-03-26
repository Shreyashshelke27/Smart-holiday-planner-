from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load the pre-saved hybrid recommendation model
try:
    with open('models/hybrid_recommendation_model.pkl', 'rb') as f:
        hybrid_model = pickle.load(f)
        
    # Extract components from the loaded model
    tfidf = hybrid_model['tfidf_vectorizer']
    tfidf_matrix = hybrid_model['tfidf_matrix']
    cosine_sim = hybrid_model['cosine_sim_matrix']
    user_item_matrix = hybrid_model['user_item_matrix']
    user_similarity = hybrid_model['user_similarity_matrix']
    merged_data = hybrid_model['merged_data']
    users_df = hybrid_model['users']
    destinations_df = hybrid_model['destinations']
    
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.route('/')
def home():
    try:
        all_preferences = set()
        for categories in destinations_df['Category'].dropna():
            all_preferences.update(cat.strip() for cat in categories.split(','))
        
        # Convert UserID to string in the dictionary
        users_list = users_df.to_dict('records')
        for user in users_list:
            user['UserID'] = str(user['UserID'])
        
        return render_template('index.html',
                             destinations=destinations_df.to_dict('records'),
                             users=users_list,
                             preferences=sorted(all_preferences),
                             states=sorted(destinations_df['State'].unique()))
    
    except Exception as e:
        print(f"Error in home route: {e}")
        return "An error occurred", 500

@app.route('/get_destination_details')
def get_destination_details():
    try:
        name = request.args.get('name')
        if not name:
            return jsonify({'error': 'Destination name is required'}), 400
            
        dest = destinations_df[destinations_df['Name'] == name]
        
        if dest.empty:
            return jsonify({'error': 'Destination not found'}), 404
            
        dest_data = dest.iloc[0]
        return jsonify({
            'District': str(dest_data['District']),
            'State': str(dest_data['State']),
            'Category': str(dest_data['Category']),
            'BestTimeToVisit': str(dest_data['BestTimeToVisit']),
            'Description': str(dest_data.get('Description', 'No description available'))
        })
        
    except Exception as e:
        print(f"Error getting destination details: {e}")
        return jsonify({'error': 'Failed to retrieve destination details'}), 500

@app.route('/get_user_details')
def get_user_details():
    try:
        user_id = request.args.get('id')
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400
        
        # Convert user_id to string for comparison
        user = users_df[users_df['UserID'].astype(str) == str(user_id)]
        
        if user.empty:
            return jsonify({'error': 'User not found'}), 404
        
        user_data = user.iloc[0]
        return jsonify({
            'Name': str(user_data['Name']),
            'Gender': str(user_data['Gender']),
            'Location': str(user_data['Location']),
            'TravelPreferences': str(user_data['TravelPreferences']),
            'NumberOfAdults': int(user_data['Number of Adults']),
            'NumberOfChildren': int(user_data['Number of Children'])
        })
        
    except Exception as e:
        print(f"Error getting user details: {e}")
        return jsonify({'error': 'Failed to retrieve user details'}), 500

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        search_type = data.get('type')
        
        if search_type == 'destination':
            destination = data.get('destination')
            if not destination:
                return jsonify({'error': 'Destination is required'}), 400
            recommendations = get_recommendations_by_destination(destination)
            
        elif search_type == 'user':
            user_id = data.get('userId')
            if not user_id:
                return jsonify({'error': 'User ID is required'}), 400
            recommendations = get_recommendations_by_user(user_id)
            
        elif search_type == 'custom':
            preferences = data.get('preferences')
            if not preferences:
                return jsonify({'error': 'Preferences are required'}), 400
            state = data.get('state')
            recommendations = get_custom_recommendations(preferences, state)
            
        else:
            return jsonify({'error': 'Invalid search type'}), 400
            
        return jsonify({'recommendations': recommendations})
        
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return jsonify({'error': str(e)}), 500

def calculate_similarity_score(query_categories, destination_categories):
    """Calculate similarity between two comma-separated category strings"""
    try:
        if not query_categories or not destination_categories:
            return 0
            
        query_cats = set(cat.strip().lower() for cat in query_categories.split(','))
        dest_cats = set(cat.strip().lower() for cat in destination_categories.split(','))
        
        if not query_cats or not dest_cats:
            return 0
        
        common_cats = query_cats.intersection(dest_cats)
        return len(common_cats) / len(query_cats) if common_cats else 0
        
    except Exception as e:
        print(f"Error calculating similarity score: {e}")
        return 0

def get_recommendations_by_destination(destination_name):
    """Use pre-loaded cosine similarity matrix to find similar destinations"""
    try:
        selected_dest = destinations_df[destinations_df['Name'] == destination_name]
        if selected_dest.empty:
            return []
            
        # Get the destination ID
        dest_id = selected_dest.iloc[0]['DestinationID']
        
        # Find similar destinations using the cosine similarity matrix
        dest_indices = merged_data[merged_data['DestinationID'] == dest_id].index
        if len(dest_indices) == 0:
            # Fallback: use TF-IDF similarity with the destination's categories
            dest_data = selected_dest.iloc[0]
            return get_custom_recommendations(dest_data['Category'])
            
        # Calculate similarity scores for all items
        sim_scores = cosine_sim[dest_indices].mean(axis=0)
        
        # Sort by similarity score in descending order
        sim_scores_with_indices = sorted([(i, score) for i, score in enumerate(sim_scores)], 
                                         key=lambda x: x[1], reverse=True)
        
        # Get top similar indices (get more to filter duplicates)
        similar_indices = [i for i, _ in sim_scores_with_indices[1:50]]  # Exclude itself
        
        recommendations = []
        seen_destinations = set([destination_name])  # Add the current destination to avoid recommending itself
        
        for idx in similar_indices:
            dest_id = merged_data.iloc[idx]['DestinationID']
            dest = destinations_df[destinations_df['DestinationID'] == dest_id]
            
            if dest.empty:
                continue
                
            dest_data = dest.iloc[0]
            dest_name = dest_data['Name']
            
            # Skip if already added
            if dest_name in seen_destinations:
                continue
                
            seen_destinations.add(dest_name)
            selected_dest_data = selected_dest.iloc[0]
            
            # Calculate category match score
            match_score = calculate_similarity_score(selected_dest_data['Category'], dest_data['Category'])
            
            # Only include destinations with at least some category overlap
            if match_score > 0:
                # Ensure the popularity score exists
                popularity_score = float(dest_data.get('PopularityScore', 0.5))
                
                recommendations.append({
                    'Name': str(dest_data['Name']),
                    'Category': str(dest_data['Category']),
                    'District': str(dest_data['District']),
                    'State': str(dest_data['State']),
                    'BestTimeToVisit': str(dest_data['BestTimeToVisit']),
                    'Description': str(dest_data.get('Description', 'No description available')),
                    'PopularityScore': popularity_score
                })
                
                if len(recommendations) >= 10:
                    break
        
        # Sort by popularity score
        recommendations.sort(key=lambda x: x['PopularityScore'], reverse=True)
        
        # Return top 10 recommendations
        return recommendations[:10]
    
    except Exception as e:
        print(f"Error in destination recommendations: {e}")
        return []

def get_recommendations_by_user(user_id):
    """Use pre-loaded model data to get recommendations for a user"""
    try:
        user_id = int(user_id)
        
        # First check if user exists
        user = users_df[users_df['UserID'] == user_id]
        if user.empty:
            return []
        
        user_data = user.iloc[0]
        user_prefs = user_data['TravelPreferences']
        
        # Check if we have preference data for this user
        if not isinstance(user_prefs, str) or not user_prefs.strip():
            # If no preferences, return popular destinations
            return get_popular_destinations()
        
        recommendations = []
        seen_destinations = set()  # Track destinations by name
        
        # Check if user exists in the model data
        if user_id in user_item_matrix.index:
            # Get user's position in user_item_matrix
            user_idx = user_item_matrix.index.get_loc(user_id)
            
            # Use user similarity to find similar users
            sim_users = user_similarity[user_idx].argsort()[::-1][1:6]  # Top 5 similar users
            
            # Get the mean item ratings from similar users
            item_scores = user_item_matrix.iloc[sim_users].mean(axis=0)
            top_items = item_scores.sort_values(ascending=False)  # Get all to filter duplicates
            
            # Convert top items to recommendations
            for dest_id, score in top_items.items():
                if score > 0:  # Only consider positive scores
                    dest = destinations_df[destinations_df['DestinationID'] == dest_id]
                    if not dest.empty:
                        dest_data = dest.iloc[0]
                        dest_name = dest_data['Name']
                        
                        # Skip if already seen
                        if dest_name in seen_destinations:
                            continue
                            
                        seen_destinations.add(dest_name)
                        
                        # Calculate match score based on user preferences
                        dest_cats = dest_data['Category']
                        match_score = calculate_similarity_score(user_prefs, dest_cats)
                        
                        # Only include if there's a category match with user preferences
                        if match_score > 0:
                            popularity_score = float(dest_data.get('PopularityScore', 0.5))
                            
                            recommendations.append({
                                'Name': str(dest_data['Name']),
                                'Category': str(dest_data['Category']),
                                'District': str(dest_data['District']),
                                'State': str(dest_data['State']),
                                'BestTimeToVisit': str(dest_data['BestTimeToVisit']),
                                'Description': str(dest_data.get('Description', 'No description available')),
                                'PopularityScore': popularity_score
                            })
                            
                            if len(recommendations) >= 10:
                                break
        
        # If we don't have enough recommendations from collaborative filtering,
        # supplement with content-based recommendations
        if len(recommendations) < 10:
            content_recommendations = get_custom_recommendations(user_prefs)
            
            # Add only new recommendations
            for rec in content_recommendations:
                if rec['Name'] not in seen_destinations and len(recommendations) < 10:
                    seen_destinations.add(rec['Name'])
                    recommendations.append(rec)
        
        # If still no recommendations, use direct custom recommendations
        if not recommendations:
            return get_custom_recommendations(user_prefs)
        
        # Sort by popularity
        recommendations.sort(key=lambda x: x['PopularityScore'], reverse=True)
        
        return recommendations[:10]
    
    except Exception as e:
        print(f"Error in user recommendations: {e}")
        return []

def get_popular_destinations():
    """Helper function to return popular destinations"""
    popular_dests = destinations_df.sort_values(by='PopularityScore', ascending=False).head(20)
    
    recommendations = []
    seen_destinations = set()
    
    for _, dest_data in popular_dests.iterrows():
        dest_name = dest_data['Name']
        
        # Skip duplicates
        if dest_name in seen_destinations:
            continue
            
        seen_destinations.add(dest_name)
        
        recommendations.append({
            'Name': str(dest_data['Name']),
            'Category': str(dest_data['Category']),
            'District': str(dest_data['District']),
            'State': str(dest_data['State']),
            'BestTimeToVisit': str(dest_data['BestTimeToVisit']),
            'Description': str(dest_data.get('Description', 'No description available')),
            'PopularityScore': float(dest_data.get('PopularityScore', 0.5))
        })
        
        if len(recommendations) >= 10:
            break
    
    return recommendations

def get_custom_recommendations(preferences, state=None):
    """Use pre-loaded TF-IDF matrix to match preferences to destinations"""
    try:
        # Handle empty preferences
        if not preferences or not preferences.strip():
            return get_popular_destinations()
        
        # Transform the preferences using the pre-trained vectorizer
        query_vector = tfidf.transform([preferences])
        
        # Calculate similarity with all items in the TF-IDF matrix
        sim_scores = (query_vector @ tfidf_matrix.T).toarray()[0]
        
        # Sort by similarity score in descending order
        sim_scores_with_indices = sorted([(i, score) for i, score in enumerate(sim_scores)], 
                                         key=lambda x: x[1], reverse=True)
        
        # Get top indices (get more to filter duplicates)
        dest_indices = [i for i, _ in sim_scores_with_indices[:50]]
        
        recommendations = []
        seen_destinations = set()  # Track destinations by name
        
        for idx in dest_indices:
            dest_id = merged_data.iloc[idx]['DestinationID']
            dest = destinations_df[destinations_df['DestinationID'] == dest_id]
            
            if dest.empty:
                continue
                
            dest_data = dest.iloc[0]
            dest_name = dest_data['Name']
            
            # Skip if already seen
            if dest_name in seen_destinations:
                continue
                
            # Apply state filter if provided
            if state and dest_data['State'] != state:
                continue
                
            # Calculate direct category match score
            match_score = calculate_similarity_score(preferences, dest_data['Category'])
            
            # Only include if there's at least some category match
            if match_score > 0:
                # Ensure popularity score exists
                popularity_score = float(dest_data.get('PopularityScore', 0.5))
                
                recommendations.append({
                    'Name': str(dest_data['Name']),
                    'Category': str(dest_data['Category']),
                    'District': str(dest_data['District']),
                    'State': str(dest_data['State']),
                    'BestTimeToVisit': str(dest_data['BestTimeToVisit']),
                    'Description': str(dest_data.get('Description', 'No description available')),
                    'PopularityScore': popularity_score
                })
                
                if len(recommendations) >= 10:
                    break
        
        # If we still don't have enough recommendations, try a more lenient approach
        if len(recommendations) < 10:
            # Split preferences into individual categories
            preference_list = [p.strip() for p in preferences.split(',')]
            
            for pref in preference_list:
                # Try to find destinations with at least one matching category
                for _, dest_data in destinations_df.iterrows():
                    dest_name = dest_data['Name']
                    
                    # Skip if already in recommendations
                    if dest_name in seen_destinations:
                        continue
                        
                    if state and dest_data['State'] != state:
                        continue
                        
                    dest_cats = dest_data['Category'].lower()
                    if pref.lower() in dest_cats:
                        seen_destinations.add(dest_name)
                        popularity_score = float(dest_data.get('PopularityScore', 0.5))
                        
                        recommendations.append({
                            'Name': str(dest_data['Name']),
                            'Category': str(dest_data['Category']),
                            'District': str(dest_data['District']),
                            'State': str(dest_data['State']),
                            'BestTimeToVisit': str(dest_data['BestTimeToVisit']),
                            'Description': str(dest_data.get('Description', 'No description available')),
                            'PopularityScore': popularity_score
                        })
                        
                        if len(recommendations) >= 10:
                            break
                            
                if len(recommendations) >= 10:
                    break
        
        # Sort by popularity score (highest popularity first)
        recommendations.sort(key=lambda x: x['PopularityScore'], reverse=True)
        
        return recommendations[:10]
    
    except Exception as e:
        print(f"Error in custom recommendations: {e}")
        return []

if __name__ == '__main__':
    app.run(debug=True, port=5000)