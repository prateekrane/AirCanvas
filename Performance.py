import matplotlib.pyplot as plt
import numpy as np

class AirCanvasPerformanceMetrics:
    def __init__(self):
        """
        Initialize the performance metrics visualization class
        """
        # Use a default matplotlib style instead of seaborn
        plt.style.use('default')

    def plot_accuracy_curve(self, training_accuracies, validation_accuracies):
        """
        Plot training and validation accuracy over epochs
        
        Parameters:
        - training_accuracies (list): Accuracy scores for training data
        - validation_accuracies (list): Accuracy scores for validation data
        """
        plt.figure(figsize=(10, 6))
        plt.plot(training_accuracies, label='Training Accuracy', color='blue', marker='o')
        plt.plot(validation_accuracies, label='Validation Accuracy', color='red', marker='s')
        plt.title('Air Canvas Model - Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, confusion_matrix, class_names):
        """
        Visualize the confusion matrix
        
        Parameters:
        - confusion_matrix (numpy.array): Confusion matrix data
        - class_names (list): Names of gesture classes
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix - Air Canvas Gesture Recognition')
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)

        # Annotate each cell with the value
        thresh = confusion_matrix.max() / 2.
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                         horizontalalignment="center",
                         verticalalignment="center",
                         color="white" if confusion_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def plot_precision_recall(self, precisions, recalls, f1_scores, class_names):
        """
        Plot precision, recall, and F1 score for each class
        
        Parameters:
        - precisions (list): Precision values for each class
        - recalls (list): Recall values for each class
        - f1_scores (list): F1 scores for each class
        - class_names (list): Names of gesture classes
        """
        plt.figure(figsize=(15, 5))
        
        # Precision subplot
        plt.subplot(1, 3, 1)
        plt.bar(class_names, precisions, color='blue', alpha=0.7)
        plt.title('Precision by Class')
        plt.xlabel('Gesture Classes')
        plt.ylabel('Precision')
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        
        # Recall subplot
        plt.subplot(1, 3, 2)
        plt.bar(class_names, recalls, color='green', alpha=0.7)
        plt.title('Recall by Class')
        plt.xlabel('Gesture Classes')
        plt.ylabel('Recall')
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        
        # F1 Score subplot
        plt.subplot(1, 3, 3)
        plt.bar(class_names, f1_scores, color='red', alpha=0.7)
        plt.title('F1 Score by Class')
        plt.xlabel('Gesture Classes')
        plt.ylabel('F1 Score')
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()

    def plot_inference_time(self, inference_times, class_names):
        """
        Plot inference times for different gestures
        
        Parameters:
        - inference_times (list): Time taken to predict each gesture
        - class_names (list): Names of gesture classes
        """
        plt.figure(figsize=(10, 6))
        plt.bar(class_names, inference_times, color='purple', alpha=0.7)
        plt.title('Inference Time by Gesture')
        plt.xlabel('Gesture Classes')
        plt.ylabel('Inference Time (ms)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# Example Usage:
def main():
    # Sample data (replace with your actual model's metrics)
    metrics = AirCanvasPerformanceMetrics()
    
    # Accuracy Curve
    training_accuracies = [0.6, 0.72, 0.80, 0.85, 0.88, 0.90, 0.92]
    validation_accuracies = [0.58, 0.68, 0.75, 0.80, 0.82, 0.84, 0.86]
    metrics.plot_accuracy_curve(training_accuracies, validation_accuracies)
    
    # Confusion Matrix
    class_names = ['Draw', 'Erase', 'Change Color', 'Select']
    confusion_matrix = np.array([
        [45, 3, 2, 0],
        [2, 48, 0, 0],
        [1, 0, 47, 2],
        [0, 1, 1, 48]
    ])
    metrics.plot_confusion_matrix(confusion_matrix, class_names)
    
    # Precision, Recall, F1 Score
    precisions = [0.90, 0.94, 0.93, 0.96]
    recalls = [0.90, 0.96, 0.94, 0.96]
    f1_scores = [0.90, 0.95, 0.93, 0.96]
    metrics.plot_precision_recall(precisions, recalls, f1_scores, class_names)
    
    # Inference Time
    inference_times = [12, 15, 10, 14]
    metrics.plot_inference_time(inference_times, class_names)

if __name__ == '__main__':
    main()