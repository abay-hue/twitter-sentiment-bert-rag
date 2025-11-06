import os, argparse, pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
from sklearn.model_selection import train_test_split

def load_csv(path):
    df = pd.read_csv(path)  # expects columns: text, label (0/1/2)
    return df

def tokenize(batch, tok):
    return tok(batch["text"], truncation=True, padding="max_length", max_length=128)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--train_csv", default="data/processed/train.csv")
    ap.add_argument("--out_dir", default="models/bert")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_csv(args.train_csv)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])
    tok = AutoTokenizer.from_pretrained(args.model)

    ds_train = Dataset.from_pandas(train_df).map(lambda b: tokenize(b, tok), batched=True)
    ds_val   = Dataset.from_pandas(val_df).map(lambda b: tokenize(b, tok), batched=True)
    cols = ["input_ids", "attention_mask", "label"]
    ds_train = ds_train.remove_columns([c for c in ds_train.column_names if c not in cols]).with_format("torch")
    ds_val   = ds_val.remove_columns([c for c in ds_val.column_names if c not in cols]).with_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(df["label"].unique()))

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        load_best_model_at_end=False,
        logging_steps=25,
        save_strategy="epoch",
        fp16=False
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=ds_train, eval_dataset=ds_val)
    trainer.train()
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("✅ Saved:", args.out_dir)

if __name__ == "__main__":
    main()
