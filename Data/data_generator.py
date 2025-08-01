import pandas as pd
import numpy as np
import random

def generate_session(user_id, is_ghost=False):
    if not is_ghost:
        return {
            "user_id": user_id,
            "device_os": random.choice(["Windows", "Android"]),
            "login_hour": int(np.random.normal(21, 1.5)),  # Normal behavior: late evening
            "typing_speed_cpm": int(np.random.normal(250, 20)),
            "nav_path": random.choice(["login>transfer", "login>balance>transfer", "login>paybill"]),
            "ip_country": "MY",
            "label": 1
        }
    else:
        return {
            "user_id": user_id,
            "device_os": random.choice(["Mac", "iOS"]),
            "login_hour": random.choice([1, 2, 3, 4]),  # Ghost sessions: early morning
            "typing_speed_cpm": int(np.random.normal(500, 30)),  # Abnormally fast
            "nav_path": random.choice(["login>settings", "login>logout"]),
            "ip_country": random.choice(["RU", "CN", "IR", "NG"]),
            "label": -1
        }

def generate_dataset(num_users=15, normal_per_user=20, ghosts_per_user=3):
    rows = []

    for uid in range(1, num_users + 1):
        user_id = f"U{uid:03d}"

        # Normal sessions
        for _ in range(normal_per_user):
            rows.append(generate_session(user_id, is_ghost=False))

        # Ghost sessions
        for _ in range(ghosts_per_user):
            rows.append(generate_session(user_id, is_ghost=True))

    df = pd.DataFrame(rows)

    # Clamp login_hour to valid range
    df["login_hour"] = df["login_hour"].clip(lower=0, upper=23)

    return df

if __name__ == "__main__":
    df = generate_dataset(num_users=15, normal_per_user=20, ghosts_per_user=4)
    df.to_csv("ghostpattern_sessions.csv", index=False)
    print("ghostpattern_sessions.csv generated successfully.")
    print(df.head())
