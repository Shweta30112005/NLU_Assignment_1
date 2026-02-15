import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# Reading documents from folder and append document as dictionary
def load_documents(folder_path, label):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()

                # keys: text, label
                documents.append({
                    "text": text,
                    "label": label
                })

    return documents


def main(project_root):

    raw_dir = os.path.join(project_root, "data", "raw")
    processed_dir = os.path.join(project_root, "data", "processed")

    os.makedirs(processed_dir, exist_ok=True)

    politics_path = os.path.join(raw_dir, "politics")
    sport_path = os.path.join(raw_dir, "sport")

    # Load raw text
    politics_docs = load_documents(politics_path, "Politics")
    sport_docs = load_documents(sport_path, "Sport")

    # Combine and shuffle data points
    all_docs = politics_docs + sport_docs

    random.seed(RANDOM_STATE)
    random.shuffle(all_docs)

    df = pd.DataFrame(all_docs)

    # Save complete dataset
    complete_path = os.path.join(processed_dir, "complete_dataset.csv")
    df.to_csv(complete_path, index=False)

    # Train-test split 
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    # Save the train and test datasets
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    print("Dataset preparation completed successfully.")
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Saved files in: {processed_dir}")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main(root)
