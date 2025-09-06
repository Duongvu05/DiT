import numpy as np
import pickle
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Random Forest Classifier for Graph Embeddings')
    parser.add_argument("--dataset", type=str, default="B4E", choices=["B4E", "MulDiGraph", "TXNT"],
                        help="Dataset to use (default: B4E)")
    parser.add_argument("--embedding_file", type=str, default=None,
                        help="Path to embedding file (default: ./dataset/<dataset>/graph_emb.txt)")
    parser.add_argument("--tag_file", type=str, default=None,
                        help="Path to tag file (default: ./dataset/<dataset>/account_tags.pkl)")
    args = parser.parse_args()

    # Set default file paths based on dataset if not provided
    if args.embedding_file is None:
        args.embedding_file = f"./dataset/{args.dataset}/graph_emb.txt"
    if args.tag_file is None:
        args.tag_file = f"./dataset/{args.dataset}/account_tags.pkl"

    return args


def get_max_index(tag_file):
    with open(tag_file, 'rb') as f:
        tags = pickle.load(f)
    return len(tags) - 1


def get_embedding_dim(embedding_file):
    with open(embedding_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                return len(parts) - 1
    return 0


def load_embeddings(file_path, max_index, embedding_dim):
    embeddings = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            index = int(parts[0])
            embedding = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            if len(embedding) == embedding_dim:
                embeddings[index] = embedding
    print(f"Number of embeddings loaded: {len(embeddings)}", end='\r')
    return embeddings


def load_tags(file_path, max_index):
    with open(file_path, 'rb') as f:
        tags = pickle.load(f)
    print(f"Number of tags loaded: {len(tags)}", end='\r')
    filtered_tags = {k: v for k, v in tags.items()}
    return filtered_tags


def prepare_data(embeddings, tags):
    valid_indices = sorted(set(embeddings.keys()) & set(tags.keys()))

    X = np.zeros((len(valid_indices), len(list(embeddings.values())[0])), dtype=np.float32)
    y = np.zeros(len(valid_indices), dtype=np.int32)

    for i, idx in enumerate(valid_indices):
        X[i] = embeddings[idx]
        y[i] = tags[idx]

    return X, y


def train_and_evaluate(X, y):
    # Chia dữ liệu: 80% train, 10% val, 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Train size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation size: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    def evaluate_split(X_split, y_split, split_name):
        y_prob = model.predict_proba(X_split)[:, 1]
        thresholds = np.arange(0.1, 1.0, 0.1)
        target_threshold = 0.3
        precision_03 = 0.0
        recall_03 = 0.0
        f1_03 = 0.0

        print(f"\n=== {split_name} Split Results ===")
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            print(f"\nThreshold: {threshold:.2f}")
            print(classification_report(y_split, y_pred, zero_division=0, digits=4))

            if abs(threshold - target_threshold) < 1e-5:
                precision_03 = precision_score(y_split, y_pred, labels=[1], average=None, zero_division=0)[0] if 1 in y_split else 0.0
                recall_03 = recall_score(y_split, y_pred, labels=[1], average=None, zero_division=0)[0] if 1 in y_split else 0.0
                f1_03 = f1_score(y_split, y_pred, labels=[1], average=None, zero_division=0)[0] if 1 in y_split else 0.0
                
                # Confusion Matrix - numpy array
                cm = confusion_matrix(y_split, y_pred)
                cm_array = np.array(cm)
                print(f"\n=== Confusion Matrix (Threshold 0.3) - {split_name} ===")
                print(f"Numpy Array:\n{cm_array}")
                
                if cm_array.shape == (2, 2):
                    tn, fp, fn, tp = cm_array.ravel()
                    print(f"True Negatives (TN): {tn}")
                    print(f"False Positives (FP): {fp}")
                    print(f"False Negatives (FN): {fn}")
                    print(f"True Positives (TP): {tp}")
                    print(f"Total samples: {tn + fp + fn + tp}")
                
                print(f"\nThreshold 0.3 - Label 1 Results:")
                print(f"Label 1 - Precision: {precision_03:.4f}")
                print(f"Label 1 - Recall: {recall_03:.4f}")
                print(f"Label 1 - F1 Score: {f1_03:.4f}")
        
        return precision_03, recall_03, f1_03

    # Đánh giá trên validation set
    val_precision, val_recall, val_f1 = evaluate_split(X_val, y_val, "Validation")
    
    # Đánh giá trên test set  
    test_precision, test_recall, test_f1 = evaluate_split(X_test, y_test, "Test")

    return (val_precision, val_recall, val_f1), (test_precision, test_recall, test_f1)


def main():
    args = parse_args()
    embedding_file = args.embedding_file
    tag_file = args.tag_file

    max_index = get_max_index(tag_file)
    print(f"Max index from tags: {max_index}")
    embedding_dim = get_embedding_dim(embedding_file)

    embeddings = load_embeddings(embedding_file, max_index, embedding_dim)
    tags = load_tags(tag_file, max_index)
    print(f"Number of tags loaded: {len(tags)}")

    X, y = prepare_data(embeddings, tags)

    (val_precision, val_recall, val_f1), (test_precision, test_recall, test_f1) = train_and_evaluate(X, y)

    print(f"\n=== Final Metrics for Label 1 (Threshold 0.3) ===")
    print(f"Validation Set:")
    print(f"  Precision: {val_precision:.4f}")
    print(f"  Recall: {val_recall:.4f}")
    print(f"  F1 Score: {val_f1:.4f}")
    
    print(f"\nTest Set:")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")


if __name__ == '__main__':
    main()