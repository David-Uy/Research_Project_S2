import os
import random
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


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
# 2. FILES
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


# =========================
# 3. LOAD DATA SAFELY
# =========================
df_list = []

for file in files:
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

print("Total samples:", len(df))
print("Loaded files:")
for f in df["source_file"].unique():
    print(" -", f)


# =========================
# 4. VALIDATE REQUIRED COLUMNS
# =========================
required_cols = BIOPHONY_COLS + ANTHROPHONY_COLS + GEOPHONY_COLS + feature_cols
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")


# =========================
# 5. SAFE LABEL CREATION
# =========================
def is_active(value):
    """
    Safely interpret label columns.
    Accepts 1, 1.0, True as active.
    Treats NaN / 0 / False as inactive.
    """
    if pd.isna(value):
        return False
    return value == 1 or value is True


def get_label(row):
    biophony = any(is_active(row[col]) for col in BIOPHONY_COLS)
    anthrophony = any(is_active(row[col]) for col in ANTHROPHONY_COLS)
    geophony = any(is_active(row[col]) for col in GEOPHONY_COLS)

    if biophony:
        return 0   # biophony
    elif anthrophony:
        return 1   # anthrophony
    elif geophony:
        return 2   # geophony
    else:
        return 3   # other


y = df.apply(get_label, axis=1).values.astype(np.int64)

# Show class distribution
unique, counts = np.unique(y, return_counts=True)
print("\nClass distribution:")
for cls, cnt in zip(unique, counts):
    print(f"Class {cls}: {cnt}")


# =========================
# 6. BUILD FEATURE MATRIX
# =========================
X = df[feature_cols].values.astype(np.float32)

print("\nNaN check before split:")
print("Total NaN values in X:", np.isnan(X).sum())
for i, col in enumerate(feature_cols):
    print(f"{col}: {np.isnan(X[:, i]).sum()}")


# =========================
# 7. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

print("\nTrain samples:", len(X_train))
print("Test samples:", len(X_test))


# =========================
# 8. HANDLE NaN + SCALE
# =========================
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

print("\nNaN check after preprocessing:")
print("NaN in X_train:", np.isnan(X_train).sum())
print("NaN in X_test:", np.isnan(X_test).sum())


# =========================
# 9. DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)


# =========================
# 10. DQN MODEL
# =========================
class DQN(nn.Module):
    def __init__(self, input_dim=13, num_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.net(x)


model = DQN(input_dim=len(feature_cols), num_actions=4).to(device)
target_model = DQN(input_dim=len(feature_cols), num_actions=4).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.SmoothL1Loss()   # more stable than MSELoss for Q-learning


# =========================
# 11. REPLAY BUFFER
# =========================
MEMORY_SIZE = 5000
BATCH_SIZE = 64
memory = deque(maxlen=MEMORY_SIZE)


def store_experience(state, action, reward, next_state, done):
    """
    Store numpy arrays, not tensors, to keep memory usage cleaner.
    """
    memory.append((state.copy(), action, reward, next_state.copy(), done))


def sample_batch():
    batch = random.sample(memory, BATCH_SIZE)

    states = np.array([b[0] for b in batch], dtype=np.float32)
    actions = np.array([b[1] for b in batch], dtype=np.int64)
    rewards = np.array([b[2] for b in batch], dtype=np.float32)
    next_states = np.array([b[3] for b in batch], dtype=np.float32)
    dones = np.array([b[4] for b in batch], dtype=np.float32)

    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    return states, actions, rewards, next_states, dones


def train_dqn_batch(gamma=0.9):
    states, actions, rewards, next_states, dones = sample_batch()

    # Current Q values
    current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Double DQN target for a bit more stability
    with torch.no_grad():
        next_actions = model(next_states).argmax(dim=1, keepdim=True)
        next_q = target_model(next_states).gather(1, next_actions).squeeze(1)
        target_q = rewards + gamma * next_q * (1 - dones)

    loss = criterion(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


# =========================
# 12. TRAINING SETTINGS
# =========================
EPISODES = 20
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.95
gamma = 0.9
target_update_freq = 2


# =========================
# 13. DQN TRAINING LOOP
# =========================
print("\n===== TRAINING DQN =====")

for episode in range(EPISODES):
    total_reward = 0.0
    correct_count = 0
    losses = []

    # Keep the order consistent with your current simplified RL setup
    # but reshuffle each episode so the agent does not always see identical order
    indices = np.random.permutation(len(X_train))
    X_train_ep = X_train[indices]
    y_train_ep = y_train[indices]

    for i in range(len(X_train_ep)):
        state = X_train_ep[i]

        # epsilon-greedy action
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = torch.argmax(model(state_tensor), dim=1).item()

        # reward
        reward = 1.0 if action == y_train_ep[i] else -1.0
        total_reward += reward

        if action == y_train_ep[i]:
            correct_count += 1

        # next state
        done = (i == len(X_train_ep) - 1)
        if done:
            next_state = np.zeros_like(state, dtype=np.float32)
        else:
            next_state = X_train_ep[i + 1]

        store_experience(state, action, reward, next_state, done)

        # train when enough experience exists
        if len(memory) >= BATCH_SIZE:
            loss = train_dqn_batch(gamma=gamma)
            losses.append(loss)

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if (episode + 1) % target_update_freq == 0:
        target_model.load_state_dict(model.state_dict())

    train_acc = correct_count / len(X_train_ep)
    avg_loss = np.mean(losses) if losses else 0.0

    print(
        f"Episode {episode + 1:02d} | "
        f"Reward: {total_reward:.1f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Avg Loss: {avg_loss:.4f} | "
        f"Epsilon: {epsilon:.2f}"
    )


# =========================
# 14. DQN EVALUATION
# =========================
print("\n===== DQN TEST RESULTS =====")
model.eval()
dqn_preds = []

with torch.no_grad():
    states_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    q_values = model(states_tensor)
    dqn_preds = torch.argmax(q_values, dim=1).cpu().numpy()

dqn_acc = accuracy_score(y_test, dqn_preds)
dqn_precision, dqn_recall, dqn_f1, _ = precision_recall_fscore_support(
    y_test, dqn_preds, average="weighted", zero_division=0
)
dqn_cm = confusion_matrix(y_test, dqn_preds)

print(f"Accuracy : {dqn_acc:.4f}")
print(f"Precision: {dqn_precision:.4f}")
print(f"Recall   : {dqn_recall:.4f}")
print(f"F1-score : {dqn_f1:.4f}")
print("\nConfusion Matrix:")
print(dqn_cm)
print("\nClassification Report:")
print(classification_report(y_test, dqn_preds, zero_division=0))


# =========================
# 15. SUPERVISED BASELINE (MLP)
# =========================
print("\n===== MLP TEST RESULTS =====")

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=SEED
)

mlp.fit(X_train, y_train)
mlp_preds = mlp.predict(X_test)

mlp_acc = accuracy_score(y_test, mlp_preds)
mlp_precision, mlp_recall, mlp_f1, _ = precision_recall_fscore_support(
    y_test, mlp_preds, average="weighted", zero_division=0
)
mlp_cm = confusion_matrix(y_test, mlp_preds)

print(f"Accuracy : {mlp_acc:.4f}")
print(f"Precision: {mlp_precision:.4f}")
print(f"Recall   : {mlp_recall:.4f}")
print(f"F1-score : {mlp_f1:.4f}")
print("\nConfusion Matrix:")
print(mlp_cm)
print("\nClassification Report:")
print(classification_report(y_test, mlp_preds, zero_division=0))


# =========================
# 16. SIMPLE COMPARISON
# =========================
print("\n===== MODEL COMPARISON =====")
print(f"DQN -> Accuracy: {dqn_acc:.4f}, F1: {dqn_f1:.4f}")
print(f"MLP -> Accuracy: {mlp_acc:.4f}, F1: {mlp_f1:.4f}")