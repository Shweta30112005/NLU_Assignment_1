from datasets import load_dataset
import os

# Loading AG News dataset
dataset = load_dataset("ag_news")

train_data = dataset["train"]

# Create folders for raw data points
os.makedirs("../data/raw/politics", exist_ok=True)
os.makedirs("../data/raw/sport", exist_ok=True)

# Counter number of sample per category
sports_count = 0
politics_count = 0
max_samples = 8000   # max number of samples per category

# Extracting label and text
for item in train_data:
    
    label = item["label"]
    text = item["text"]

    # AG News Dataset has 4 classes
    # World/Politics (0), Sports(1), Business(2), Science(3)

    # extracting only politics and sport
    
    if label == 0 and politics_count < max_samples:
        filename = f"../data/raw/politics/politics_{politics_count}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        politics_count += 1

    elif label == 1 and sports_count < max_samples:
        filename = f"../data/raw/sport/sport_{sports_count}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        sports_count += 1

    if sports_count >= max_samples and politics_count >= max_samples:
        break

print("Dataset Generated Successfully")
