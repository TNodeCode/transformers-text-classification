import click
import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
from transformers import TextClassificationPipeline
from datasets import load_dataset, Dataset
import os


@click.group()
def cli():
    pass


@cli.command()
@click.option('--epochs', default=3, help='Number of training epochs')
@click.option('--output_dir', default='work_dirs', help='Directory to save the trained model')
@click.option('--train_csv', default='./data/imdb/imdb_train.csv', help='Path to the training CSV file')
@click.option('--val_csv', default='./data/imdb/imdb_test.csv', help='Path to the validation CSV file')
@click.option('--fine_tune', is_flag=True, default=False, help='If not set, only the prediction head will be trained')
@click.option('--random_init', is_flag=True, default=False, help='If set, initializes model with random weights instead of pretrained')
def train(epochs, output_dir, train_csv, val_csv, fine_tune, random_init):
    """Train a DistilBERT model on a dataset loaded from CSV files with validation after each epoch."""
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Load and preprocess datasets from CSV files
    def load_and_tokenize(file_path):
        df = pd.read_csv(file_path)
        num_classes = df['label'].nunique()
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True), batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        return dataset, num_classes

    print("Tokenizing training data ...")
    train_dataset, num_classes_train = load_and_tokenize(train_csv)
    print("Tokenizing validation data ...")
    val_dataset, num_classes_val = load_and_tokenize(val_csv)

    assert num_classes_train == num_classes_val, f"Number of classes in training and testing data should be equal. Got {num_classes_train} classes for the training data and {num_classes_val} for the testing data."

    # Initialize model with or without pretrained weights
    model_init = "distilbert-base-uncased" if not random_init else None
    model = DistilBertForSequenceClassification.from_pretrained(
        model_init or "distilbert-base-uncased",  # Use pretrained or a new random initialization
        num_labels=num_classes_train,
    )

    if random_init:
        # If random initialization is selected, reset weights
        model.init_weights()    # Set up training arguments and Trainer

    # Freeze all layers except the prediction head if training head only
    if not fine_tune:
        for param in model.distilbert.parameters():
            param.requires_grad = False
        print("Only the prediction head will be trained.")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",  # Evaluate after every epoch
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train model with evaluation
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model trained and saved in directory {output_dir}")


@cli.command()
@click.option('--model_dir', default='work_dirs', help='Directory of the trained model')
def test(model_dir):
    """Evaluate the model on the test dataset."""
    # Load dataset and tokenizer
    dataset = load_dataset("imdb")
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)

    # Tokenize test dataset
    test_dataset = dataset["test"].map(lambda x: tokenizer(x["text"], padding=True, truncation=True), batched=True)
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Load model
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)

    # Set up Trainer for evaluation
    training_args = TrainingArguments(
        output_dir=model_dir,
        per_device_eval_batch_size=16,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
    )

    # Evaluate model
    results = trainer.evaluate()
    click.echo(f"Test evaluation results: {results}")


@cli.command()
@click.argument('input_csv', type=click.Path(exists=True))
@click.argument('output_csv', type=click.Path())
@click.option('--model_dir', default='work_dirs', help='Directory of the trained model')
def inference(input_csv, output_csv, model_dir):
    """Run inference on a CSV file of sentences and save results."""
    # Load model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=-1)

    # Load input data
    data = pd.read_csv(input_csv)
    if 'sentence' not in data.columns:
        raise ValueError("The input CSV must contain a 'sentence' column")

    # Run inference
    data['label'] = data['sentence'].apply(lambda x: pipeline(x)[0]['label'])

    # Save results
    data.to_csv(output_csv, index=False)
    click.echo(f"Inference results saved in {output_csv}")


@cli.command()
@click.argument('output_dir', type=click.Path())
@click.option('--model_dir', default='work_dirs', help='Directory of the trained model')
def export_onnx(output_dir, model_dir):
    """Export the trained model to ONNX format."""
    # Load model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)

    # Export to ONNX
    dummy_input = tokenizer("dummy input", return_tensors="pt")
    onnx_path = os.path.join(output_dir, "distilbert_model.onnx")
    torch.onnx.export(model, (dummy_input['input_ids'], dummy_input['attention_mask']), onnx_path,
                      input_names=['input_ids', 'attention_mask'],
                      output_names=['output'],
                      dynamic_axes={'input_ids': {0: 'batch_size'}, 'attention_mask': {0: 'batch_size'}})
    click.echo(f"Model exported to ONNX format at {onnx_path}")


if __name__ == '__main__':
    cli()