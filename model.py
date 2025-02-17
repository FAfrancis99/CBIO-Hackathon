import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from model_selection import select_best_model, evaluate_final_model


def parse_fasta_file(file_path: str, label):
    sequences = {}
    with open(file_path, 'r') as file_handle:
        for record in SeqIO.parse(file_handle, "fasta"):
            sequences[str(record.seq)] = label
    return sequences


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)

    # Load data
    antibiotic = parse_fasta_file("BLUE/betalactamases_antibiotic_resistant.fa", True)
    non_antibiotic = parse_fasta_file("BLUE/betalactamases_non-antibiotic_resistant.fa", False)
    #Aminoglycosides_antibiotic_resistant.fa
    # antibiotic = parse_fasta_file("RED/Aminoglycosides_antibiotic_resistant.fa", True)
    # non_antibiotic = parse_fasta_file("RED/Aminoglycosides_non-antibiotic_resistant.fa", False)
    print(len(antibiotic), len(non_antibiotic))

    # Combine all sequences and labels
    all_sequences = list(antibiotic.keys()) + list(non_antibiotic.keys())
    all_labels = list(antibiotic.values()) + list(non_antibiotic.values())

    # Create a single train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        all_sequences,
        all_labels,
        test_size=0.2,
        random_state=42,
        stratify=all_labels  # Ensure balanced split
    )

    # Create labels dictionaries for training and test data
    train_labels = dict(zip(X_train, y_train))
    test_labels = dict(zip(X_test, y_test))

    # Step 1: Select best model using only training data
    best_config, cv_results = select_best_model(
        hyperparameters=[2,3,4,5],
        train_data=X_train,
        train_labels=train_labels
    )

    # Step 2: Evaluate final model on test set
    final_results = evaluate_final_model(
        best_config,
        X_train,
        train_labels,
        X_test,
        test_labels
    )

    # Print dataset statistics
    print("\nDataset statistics:")
    print(f"Total sequences: {len(all_sequences)}")
    print(f"Antibiotic resistant: {sum(all_labels)}")
    print(f"Non-antibiotic resistant: {len(all_labels) - sum(all_labels)}")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Final best model is available in final_results['model']