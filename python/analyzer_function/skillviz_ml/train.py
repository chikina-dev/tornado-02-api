
"""
(ML Version) Trains a model on the extracted features.
This example uses unsupervised clustering to identify user personas.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    """Loads features, trains a clustering model, and analyzes results."""
    parser = argparse.ArgumentParser(description="Train a model on skill features.")
    parser.add_argument("--features", type=Path, required=True, help="Path to the feature matrix (ml_features.parquet)")
    parser.add_argument("--n-clusters", type=int, default=4, help="Number of user personas (clusters) to identify")
    args = parser.parse_args()

    print(f"Loading feature matrix from {args.features}...")
    try:
        df = pd.read_parquet(args.features)
    except Exception as e:
        print(f"Failed to load feature file: {e}")
        return

    # --- Feature Preparation ---
    # Pivot table to get one row per user, with skills as columns
    df_pivot = df.pivot_table(
        index='user_id',
        columns='skill',
        values='heuristic_score_sum', # Using this score as the main value for pivoting
        fill_value=0
    )

    # Also aggregate other features across all skills for each user
    user_features = df.groupby('user_id').agg(
        mean_difficulty_overall=('mean_difficulty', 'mean'),
        mean_engagement_overall=('mean_engagement', 'mean'),
        n_pages_overall=('n_pages', 'sum')
    )
    
    # Combine skill matrix with aggregated user features
    df_model_input = pd.concat([df_pivot, user_features], axis=1)

    print("\n--- Model Input Features ---")
    print(df_model_input.head())

    # Scale features for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_model_input)

    # --- Model Training (KMeans Clustering) ---
    print(f"\nTraining KMeans model with {args.n_clusters} clusters...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    df_model_input['persona_cluster'] = kmeans.labels_

    # --- Persona Analysis ---
    print("\n--- Identified User Personas ---")
    for i in range(args.n_clusters):
        print(f"\n[ Persona Cluster {i} ]")
        cluster_users = df_model_input[df_model_input['persona_cluster'] == i]
        print(f"  - Users in this persona: {list(cluster_users.index)}")
        
        # Analyze the characteristics of this cluster
        persona_profile = cluster_users.drop(columns=['persona_cluster']).mean()
        top_skills = persona_profile[df_pivot.columns].nlargest(3)

        print("  - Top 3 Skills:")
        for skill, score in top_skills.items():
            print(f"    - {skill} (Avg Score: {score:.2f})")
        
        print("  - General Characteristics:")
        print(f"    - Overall Average Difficulty: {persona_profile['mean_difficulty_overall']:.2f}")
        print(f"    - Overall Average Engagement: {persona_profile['mean_engagement_overall']:.2f}")

if __name__ == "__main__":
    main()
