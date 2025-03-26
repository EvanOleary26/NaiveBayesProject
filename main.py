
from DataLoader import DataPreprocessing
from NaiveBayes import NBClassifier
from EvaluationMetrics import EvaluationMetrics

def main():
    # Step 1: Load and preprocess the data
    data_loader = DataPreprocessing()
    raw_data = data_loader.load_data()
    training_data, testing_data = data_loader.split_data(raw_data)

    print(f"Training data size: {len(training_data)}")
    print(f"Testing data size: {len(testing_data)}")

    # Step 2: Train the Naive Bayes classifier
    classifier = NBClassifier()
    classifier.train(training_data)

    # Step 3: Make predictions on the testing data
    test_messages = [message for _, message in testing_data]
    actual_labels = [label for label, _ in testing_data]
    predicted_labels = classifier.prediction(test_messages)

    # Step 4: Evaluate the model
    evaluator = EvaluationMetrics()
    metrics = evaluator.compute_metrics(actual_labels, predicted_labels)

    # Step 5: Log the results
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")

    # Optional: Save results to a file
    with open("results.log", "w") as log_file:
        log_file.write("Evaluation Metrics:\n")
        for metric, value in metrics.items():
            log_file.write(f"{metric}: {value:.4f}\n" if isinstance(value, float) else f"{metric}: {value}\n")



if __name__ == "__main__":
    main()