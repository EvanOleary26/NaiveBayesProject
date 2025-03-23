
class EvaluationMetrics:
        def compute_metrics(self, actual_labels, predicted_labels):
            '''
            compute_metrics():
            - compute True Positives, True Negatives, False Positives, False Negatives
            - compute accuracy, precision, recall, F1 score
            '''
            # Initialize counts
            TP = TN = FP = FN = 0

            # Calculate TP, TN, FP, FN
            for actual, predicted in zip(actual_labels, predicted_labels):
                if actual == 1 and predicted == 1:
                    TP += 1
                elif actual == 0 and predicted == 0:
                    TN += 1
                elif actual == 0 and predicted == 1:
                    FP += 1
                elif actual == 1 and predicted == 0:
                    FN += 1

            # Compute metrics
            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Return metrics as a dictionary
            return {
                "True Positives": TP,
                "True Negatives": TN,
                "False Positives": FP,
                "False Negatives": FN,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1_score
            }