import os
import random
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.cluster import KMeans, DBSCAN
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight


# =========================
# 1. REPRODUCIBILITY
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =========================
# 2. FILES / CONFIG
# =========================
files = [
    "DUVAL-DRYA-20210419T000000+1000_REC_annotations_MFCC.csv",
    "MOURACHAN-WETA-20210509T000000+1000_REC_annotations_MFCC.csv",
    "RINYIRRU-WETB-20210615T080000+1000_REC_annotations_MFCC.csv",
    "UNDARA-DRYB-20210604T080000+1000_REC_annotations_MFCC.csv"
]

feature_cols = [f"mfcc_{i}" for i in range(1, 14)]

BIOPHONY_COLS = ["birds", "frogs", "insects", "mammals"]
ANTHROPHONY_COLS = ["human_speech", "vehicles_(aircraft/cars)"]
GEOPHONY_COLS = ["rain_(heavy)", "rain_(light)", "wind_(strong)", "wind_(light)"]

CLASS_NAMES = {
    0: "biophony",
    1: "anthrophony",
    2: "geophony",
    3: "other"
}


# =========================
# 3. LOAD DATA
# =========================
def load_data(file_list):
    df_list = []

    for file in file_list:
        if not os.path.exists(file):
            print(f"[WARNING] File not found: {file}")
            continue

        temp_df = pd.read_csv(file)
        temp_df["source_file"] = os.path.basename(file)
        df_list.append(temp_df)

    if not df_list:
        raise FileNotFoundError("No valid CSV files were found. Check your file paths.")

    df = pd.concat(df_list, ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


def validate_columns(df):
    required_cols = BIOPHONY_COLS + ANTHROPHONY_COLS + GEOPHONY_COLS + feature_cols
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


# =========================
# 4. LABEL CREATION
# =========================
def is_active(value):
    if pd.isna(value):
        return False
    return value == 1 or value is True


def get_label(row):
    biophony = any(is_active(row[col]) for col in BIOPHONY_COLS)
    anthrophony = any(is_active(row[col]) for col in ANTHROPHONY_COLS)
    geophony = any(is_active(row[col]) for col in GEOPHONY_COLS)

    if biophony:
        return 0
    elif anthrophony:
        return 1
    elif geophony:
        return 2
    else:
        return 3


# =========================
# 5. PREPROCESSING
# =========================
def prepare_dataset(df):
    y = df.apply(get_label, axis=1).values.astype(np.int64)
    X = df[feature_cols].values.astype(np.float32)

    print("Total samples:", len(df))

    print("\nLoaded files:")
    for f in df["source_file"].unique():
        print(" -", f)

    print("\nOriginal class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"Class {cls} ({CLASS_NAMES[cls]}): {cnt}")

    print("\nNaN check before split:")
    print("Total NaN values in X:", np.isnan(X).sum())

    for i, col in enumerate(feature_cols):
        print(f"{col}: {np.isnan(X[:, i]).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    print("\nTrain samples:", len(X_train))
    print("Test samples:", len(X_test))

    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    print("\nNaN check after preprocessing:")
    print("NaN in X_train:", np.isnan(X_train).sum())
    print("NaN in X_test:", np.isnan(X_test).sum())

    return X_train, X_test, y_train, y_test


# =========================
# 6. OVERSAMPLING
# =========================
def oversample_minority_classes(X_train, y_train):
    unique_classes, counts = np.unique(y_train, return_counts=True)
    max_count = counts.max()

    X_balanced = []
    y_balanced = []

    for cls in unique_classes:
        cls_indices = np.where(y_train == cls)[0]

        sampled_indices = np.random.choice(
            cls_indices,
            size=max_count,
            replace=True
        )

        X_balanced.append(X_train[sampled_indices])
        y_balanced.append(y_train[sampled_indices])

    X_balanced = np.vstack(X_balanced).astype(np.float32)
    y_balanced = np.concatenate(y_balanced).astype(np.int64)

    shuffle_idx = np.random.permutation(len(y_balanced))

    return X_balanced[shuffle_idx], y_balanced[shuffle_idx]


# =========================
# 7. DQN MODEL
# =========================
class DQN(nn.Module):
    def __init__(self, input_dim=13, num_actions=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.net(x)


class DQNTrainer:
    def __init__(self, input_dim, num_actions, device):
        self.device = device
        self.num_actions = num_actions

        self.model = DQN(input_dim=input_dim, num_actions=num_actions).to(device)
        self.target_model = DQN(input_dim=input_dim, num_actions=num_actions).to(device)

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.SmoothL1Loss()

        self.memory = deque(maxlen=10000)
        self.batch_size = 64

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state.copy(), action, reward, next_state.copy(), done))

    def sample_batch(self):
        batch = random.sample(self.memory, self.batch_size)

        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones

    def train_batch(self, gamma=0.9):
        states, actions, rewards, next_states, dones = self.sample_batch()

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_model(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + gamma * next_q * (1 - dones)

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def fit(
        self,
        X_train,
        y_train,
        episodes=20,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.95,
        target_update_freq=2
    ):
        print("\n===== TRAINING BALANCED DQN =====")

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train
        )

        reward_weights = {
            cls: weight for cls, weight in zip(np.unique(y_train), class_weights)
        }

        print("\nDQN reward weights:")
        for cls, weight in reward_weights.items():
            print(f"Class {cls} ({CLASS_NAMES[cls]}): {weight:.4f}")

        for episode in range(episodes):
            total_reward = 0.0
            correct_count = 0
            losses = []

            indices = np.random.permutation(len(X_train))
            X_train_ep = X_train[indices]
            y_train_ep = y_train[indices]

            for i in range(len(X_train_ep)):
                state = X_train_ep[i]
                true_class = y_train_ep[i]

                if random.random() < epsilon:
                    action = random.randint(0, self.num_actions - 1)
                else:
                    with torch.no_grad():
                        state_tensor = torch.tensor(
                            state,
                            dtype=torch.float32,
                            device=self.device
                        ).unsqueeze(0)

                        action = torch.argmax(self.model(state_tensor), dim=1).item()

                if action == true_class:
                    reward = float(reward_weights[true_class])
                    correct_count += 1
                else:
                    reward = -0.5

                total_reward += reward

                done = (i == len(X_train_ep) - 1)

                if done:
                    next_state = np.zeros_like(state, dtype=np.float32)
                else:
                    next_state = X_train_ep[i + 1]

                self.store_experience(state, action, reward, next_state, done)

                if len(self.memory) >= self.batch_size:
                    loss = self.train_batch(gamma=gamma)
                    losses.append(loss)

            epsilon = max(epsilon * epsilon_decay, epsilon_min)

            if (episode + 1) % target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            train_acc = correct_count / len(X_train_ep)
            avg_loss = np.mean(losses) if losses else 0.0

            print(
                f"Episode {episode + 1:02d} | "
                f"Reward: {total_reward:.2f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Epsilon: {epsilon:.2f}"
            )

    def predict(self, X):
        self.model.eval()

        with torch.no_grad():
            states_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            q_values = self.model(states_tensor)
            preds = torch.argmax(q_values, dim=1).cpu().numpy()

        return preds


# =========================
# 8. EVALUATION HELPERS
# =========================
def evaluate_classifier(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    print(f"\n===== {name} TEST RESULTS =====")
    print(f"Accuracy           : {acc:.4f}")
    print(f"Balanced Accuracy  : {balanced_acc:.4f}")
    print(f"Weighted Precision : {weighted_precision:.4f}")
    print(f"Weighted Recall    : {weighted_recall:.4f}")
    print(f"Weighted F1-score  : {weighted_f1:.4f}")
    print(f"Macro Precision    : {macro_precision:.4f}")
    print(f"Macro Recall       : {macro_recall:.4f}")
    print(f"Macro F1-score     : {macro_f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {
        "model": name,
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }


def evaluate_clustering(name, y_true, cluster_labels):
    valid_mask = cluster_labels != -1

    if valid_mask.sum() == 0:
        ari = 0.0
        nmi = 0.0
        coverage = 0.0
    else:
        ari = adjusted_rand_score(y_true[valid_mask], cluster_labels[valid_mask])
        nmi = normalized_mutual_info_score(y_true[valid_mask], cluster_labels[valid_mask])
        coverage = valid_mask.mean()

    print(f"\n===== {name} CLUSTERING RESULTS =====")
    print(f"ARI      : {ari:.4f}")
    print(f"NMI      : {nmi:.4f}")
    print(f"Coverage : {coverage:.4f}")

    return {
        "model": name,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "weighted_precision": np.nan,
        "weighted_recall": np.nan,
        "weighted_f1": np.nan,
        "macro_precision": np.nan,
        "macro_recall": np.nan,
        "macro_f1": np.nan,
        "ari": ari,
        "nmi": nmi,
        "coverage": coverage
    }


# =========================
# 9. SUPERVISED MODELS
# =========================
def run_supervised_models(X_train, X_test, y_train, y_test):
    models = {
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=SEED
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=SEED,
            class_weight="balanced_subsample"
        ),

        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=SEED
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=7
        ),

        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=SEED,
            class_weight="balanced"
        ),

        "DummyMostFrequent": DummyClassifier(
            strategy="most_frequent",
            random_state=SEED
        ),

        "DummyStratified": DummyClassifier(
            strategy="stratified",
            random_state=SEED
        )
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics_row = evaluate_classifier(name, y_test, preds)
        results.append(metrics_row)

    return results


# =========================
# 10. UNSUPERVISED MODELS
# =========================
def run_clustering_models(X_train, X_test, y_test):
    results = []

    kmeans = KMeans(n_clusters=4, random_state=SEED, n_init=10)
    kmeans.fit(X_train)
    kmeans_labels = kmeans.predict(X_test)
    results.append(evaluate_clustering("KMeans", y_test, kmeans_labels))

    dbscan = DBSCAN(eps=1.2, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X_test)
    results.append(evaluate_clustering("DBSCAN", y_test, dbscan_labels))

    return results


# =========================
# 11. MAIN
# =========================
def main():
    df = load_data(files)
    validate_columns(df)

    X_train, X_test, y_train, y_test = prepare_dataset(df)

    X_train_balanced, y_train_balanced = oversample_minority_classes(
        X_train,
        y_train
    )

    print("\nBalanced training distribution:")
    unique, counts = np.unique(y_train_balanced, return_counts=True)

    for cls, cnt in zip(unique, counts):
        print(f"Class {cls} ({CLASS_NAMES[cls]}): {cnt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)

    all_results = []

    # =========================
    # DQN
    # =========================
    dqn_trainer = DQNTrainer(
        input_dim=len(feature_cols),
        num_actions=4,
        device=device
    )

    dqn_trainer.fit(
        X_train_balanced,
        y_train_balanced,
        episodes=20
    )

    dqn_preds = dqn_trainer.predict(X_test)
    dqn_result = evaluate_classifier("Balanced DQN", y_test, dqn_preds)
    all_results.append(dqn_result)

    # =========================
    # SUPERVISED MODELS
    # =========================
    supervised_results = run_supervised_models(
        X_train_balanced,
        X_test,
        y_train_balanced,
        y_test
    )

    all_results.extend(supervised_results)

    # =========================
    # UNSUPERVISED MODELS
    # =========================
    clustering_results = run_clustering_models(
        X_train,
        X_test,
        y_test
    )

    all_results.extend(clustering_results)

    # =========================
    # FINAL COMPARISON
    # =========================
    results_df = pd.DataFrame(all_results)

    print("\n===== FINAL MODEL COMPARISON =====")
    print(results_df)

    results_df.to_csv("balanced_model_comparison_results.csv", index=False)

    print("\nSaved results to: balanced_model_comparison_results.csv")


if __name__ == "__main__":
    main()