import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# =========================
# 1. LOAD DATA
# =========================
files = [
    "DUVAL-DRYA-20210419T000000+1000_REC_annotations_MFCC.csv",
    "MOURACHAN-WETA-20210509T000000+1000_REC_annotations_MFCC.csv",
    "RINYIRRU-WETB-20210615T080000+1000_REC_annotations_MFCC.csv",
    "UNDARA-DRYB-20210604T080000+1000_REC_annotations_MFCC.csv"
]

df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)

# shuffle data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Total samples:", len(df))
# MFCC features
X = df[[f"mfcc_{i}" for i in range(1, 14)]].values.astype(np.float32)

# =========================
# 2. CREATE LABELS (4 classes)
# =========================
def get_label(row):
    if row['birds'] or row['frogs'] or row['insects'] or row['mammals']:
        return 0  # biophony
    elif row['human_speech'] or row['vehicles_(aircraft/cars)']:
        return 1  # anthrophony
    elif row['rain_(heavy)'] or row['rain_(light)'] or row['wind_(strong)'] or row['wind_(light)']:
        return 2  # geophony
    else:
        return 3  # other

y = df.apply(get_label, axis=1).values

# =========================
# 3. DQN MODEL
# =========================
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)

model = DQN()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

# =========================
# 4. REPLAY BUFFER
# =========================
memory = []
MEMORY_SIZE = 1000
BATCH_SIZE = 32

def store_experience(exp):
    if len(memory) >= MEMORY_SIZE:
        memory.pop(0)
    memory.append(exp)

def sample_batch():
    return random.sample(memory, min(len(memory), BATCH_SIZE))

# =========================
# 5. TRAINING SETTINGS
# =========================
EPISODES = 20
epsilon = 1.0       # exploration
epsilon_min = 0.1
epsilon_decay = 0.98
gamma = 0.9         # discount

# =========================
# 6. TRAIN LOOP
# =========================
for episode in range(EPISODES):
    total_reward = 0

    for i in range(len(X)):
        state = torch.tensor(X[i])

        # epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                action = torch.argmax(model(state)).item()

        # reward
        reward = 2 if action == y[i] else -1
        total_reward += reward

        # next state (simple: next row)
        next_state = torch.tensor(X[(i+1) % len(X)])

        # store experience
        store_experience((state, action, reward, next_state))

        # =========================
        # TRAIN FROM MEMORY
        # =========================
        batch = sample_batch()

        for s, a, r, ns in batch:
            q_values = model(s)
            with torch.no_grad():
                next_q = torch.max(model(ns))

            target = q_values.clone().detach()
            target[a] = r + gamma * next_q

            loss = criterion(q_values, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # decay epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    print(f"Episode {episode+1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

# =========================
# 7. SIMPLE EVALUATION
# =========================
correct = 0

with torch.no_grad():
    for i in range(len(X)):
        state = torch.tensor(X[i])
        pred = torch.argmax(model(state)).item()
        if pred == y[i]:
            correct += 1

accuracy = correct / len(X)
print(f"\nFinal Accuracy: {accuracy:.4f}")