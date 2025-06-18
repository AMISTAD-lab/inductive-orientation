# """Seeing if fine-tuning SBERT is something we can do"""
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# from datasets import load_dataset
# import torch

# # Load TinyBERT
# model_name = "huawei-noah/TinyBERT_General_4L_312D"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Load model with classification head (e.g., for 2 classes)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# # Load and preprocess a sample dataset (you can use your own)
# dataset = load_dataset("imdb")  # Binary sentiment classification
# def tokenize_fn(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)

# tokenized_dataset = dataset.map(tokenize_fn, batched=True)
# tokenized_dataset = tokenized_dataset.rename_column("label", "labels")  # Required for Trainer
# tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# # Training configuration
# training_args = TrainingArguments(
#     output_dir="./tinybert-classifier",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=10,
#     load_best_model_at_end=True,
# )

# # Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"].shuffle(seed=42).select(range(5000)),  # for quick demo
#     eval_dataset=tokenized_dataset["test"].select(range(1000)),
# )

# # Train the model
# trainer.train()
