import numpy as np
import pandas as pd
import os
import gc
from tqdm import tqdm
from sklearn.metrics import accuracy_score, adjusted_rand_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from Config import p_pool_size, n_PTRs
from Utils import Pattern_pool, match_ultimate
import pickle

# Initialize the pattern pool
p_pool = Pattern_pool(p_pool_size)

# File to track progress and performance
checkpoint_file = 'progress_checkpoint.txt'
metrics_file = 'performance_metrics.csv'

# Function to manually oversample small clusters in the training set
def manual_oversampling(df, target_col='bi_einstaklingur', min_size=20):
    """Manually oversample smaller clusters to ensure a minimum cluster size."""
    cluster_counts = df[target_col].value_counts()
    oversampled_data = pd.DataFrame()
    
    print("Oversampling small clusters...")
    for cluster, count in tqdm(cluster_counts.items(), desc="Oversampling clusters", ncols=100):
        cluster_data = df[df[target_col] == cluster]
        if count < min_size:
            # Oversample small clusters
            oversampled_cluster_data = cluster_data.sample(min_size, replace=True, random_state=42)
        else:
            oversampled_cluster_data = cluster_data  # Keep large clusters as is

        oversampled_data = pd.concat([oversampled_data, oversampled_cluster_data])

    return oversampled_data

# Split the dataset into training and testing based on bi_einstaklingur clusters
def split_train_test(df, train_size=0.6, test_size=0.2, compare_size=0.2):
    """Split the dataset according to the knobs into training, testing, and comparison sets."""
    assert train_size + test_size + compare_size == 1, "The sum of train, test, and compare sizes must be 1."

    # Shuffle and split by unique clusters
    unique_clusters = df['bi_einstaklingur'].unique()
    np.random.shuffle(unique_clusters)

    train_cutoff = int(len(unique_clusters) * train_size)
    test_cutoff = int(len(unique_clusters) * (train_size + test_size))

    train_clusters = unique_clusters[:train_cutoff]
    test_clusters = unique_clusters[train_cutoff:test_cutoff]
    compare_clusters = unique_clusters[test_cutoff:]

    # Extract training, testing, and comparison data
    train_df = df[df['bi_einstaklingur'].isin(train_clusters)]
    test_df = df[df['bi_einstaklingur'].isin(test_clusters)]
    compare_df = df[df['bi_einstaklingur'].isin(compare_clusters)]

    return train_df, test_df, compare_df

# Function to perform parallel comparisons for a chunk of rows using match_ultimate
def compare_row_with_others(idx, row, df_np_chunk, p_pool, eval_only=False):
    matching_ids = []
    for compare_idx, r2 in enumerate(df_np_chunk):
        if idx != compare_idx:
            match_score = match_ultimate(row, r2, p_pool, n_PTRs, just_eval=eval_only)
            if match_score > 0.5:  # Assuming a threshold of 0.5 for matching
                matching_ids.append(r2[0])  # Add the row ID of the matching row
    return matching_ids

# Function to perform model evaluation on 10% of the test set at a time
def evaluate_in_chunks(test_df, chunk_percentage=0.1):
    total_rows = len(test_df)
    chunk_size = int(total_rows * chunk_percentage)
    num_chunks = total_rows // chunk_size + (1 if total_rows % chunk_size != 0 else 0)
    
    total_accuracy, total_ari = 0, 0
    num_evaluated = 0
    
    for chunk_idx in tqdm(range(num_chunks), desc="Evaluating chunks", ncols=100):
        start_row = chunk_idx * chunk_size
        end_row = min(start_row + chunk_size, total_rows)
        
        df_chunk = test_df.iloc[start_row:end_row]
        true_clusters = df_chunk['bi_einstaklingur'].tolist()
        
        # Perform comparisons for the current chunk
        predicted_ids = run_comparisons_chunk(df_chunk.to_numpy(), eval_only=True)
        df_chunk['nars_predicted'] = predicted_ids
        
        # Compute metrics for this chunk
        true_pairs, pred_pairs = [], []
        for i in range(len(df_chunk)):
            row_true_cluster = true_clusters[i]
            row_pred_cluster = df_chunk['nars_predicted'][i]
            for j in range(i + 1, len(df_chunk)):
                true_pairs.append(1 if true_clusters[j] == row_true_cluster else 0)
                pred_pairs.append(1 if df_chunk.index[j] in row_pred_cluster else 0)
        
        if true_pairs and pred_pairs:
            accuracy = accuracy_score(true_pairs, pred_pairs)
            ari = adjusted_rand_score(true_pairs, pred_pairs)
            total_accuracy += accuracy
            total_ari += ari
            num_evaluated += 1

            print(f"Chunk {chunk_idx + 1}/{num_chunks} | Accuracy: {accuracy:.4f}, ARI: {ari:.4f}")
        
        # Free memory after each chunk
        del df_chunk
        gc.collect()

    # Compute average metrics after processing all chunks
    if num_evaluated > 0:
        avg_accuracy = total_accuracy / num_evaluated
        avg_ari = total_ari / num_evaluated
        print(f"Overall Evaluation | Average Accuracy: {avg_accuracy:.4f}, Average ARI: {avg_ari:.4f}")
        save_metrics_to_file(avg_accuracy, avg_ari, 'final')

# Function to perform parallel comparisons for a chunk of rows
def run_comparisons_chunk(df_np_chunk, eval_only=False, max_workers=4):
    predicted_ids = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(compare_row_with_others, idx, row, df_np_chunk, p_pool, eval_only) for idx, row in enumerate(df_np_chunk)
        ]
        for future in tqdm(as_completed(futures), desc="Comparing rows", ncols=100, total=len(futures)):
            predicted_ids.append(future.result())
    return predicted_ids

# Save classical models to disk
def save_model(model, model_name):
    with open(f'saved_models/{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"Model {model_name} saved successfully.")

# Function to save metrics to a file (append metrics after final evaluation)
def save_metrics_to_file(accuracy, ari, row_num):
    try:
        with open(metrics_file, 'a') as f:  # Append to existing file
            f.write(f"{row_num},{accuracy:.4f},{ari:.4f}\n")
        print(f"Metrics saved. Accuracy: {accuracy:.4f}, ARI: {ari:.4f}")
    except Exception as e:
        print(f"Error writing to metrics file: {e}")

# Main function
def main():
    print("Running main...")

    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('./rule_based_predictions.csv')
    df.reset_index(drop=True, inplace=True)
    print(f"Dataset loaded with {len(df)} rows.")

    # Drop unnecessary columns
    df.drop(["fornafn", "millinafn", "eftirnafn", "aettarnafn"], axis=1, inplace=True, errors='ignore')
    print(f"Dataset loaded with {len(df)} rows after dropping unnecessary columns.")

    # Split the dataset into training, testing, and comparison sets
    print("Splitting dataset...")
    global train_df, test_df, compare_df
    train_df, test_df, compare_df = split_train_test(df, train_size=0.6, test_size=0.2, compare_size=0.2)

    # Manually oversample the training data
    print("Performing manual oversampling on the training set...")
    train_df_balanced = manual_oversampling(train_df)
    print(f"Training set balanced. {len(train_df_balanced)} rows after oversampling.")

    # Train the model on the balanced training dataset
    print("Training model on the balanced training dataset...")
    df_np_train = train_df_balanced.to_numpy()
    for idx, row in tqdm(enumerate(df_np_train), desc="Training", ncols=100):
        compare_row_with_others(idx, row, df_np_train, p_pool, eval_only=False)
    
    print("Training complete.")

    # Evaluate on 10% of the test set at a time
    print("Evaluating the model on the test set in chunks...")
    evaluate_in_chunks(test_df, chunk_percentage=0.1)

    # Save the final trained model
    save_model(p_pool, "trained_pattern_pool_final.pkl")
    print("Model saved.")
    test_df.to_csv('./predicted_test_results.csv', index=False)
    print("Test results saved.")

if __name__ == "__main__":
    main()