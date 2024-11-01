ALGORITHM: Air Canvas Performance Metrics Visualization

MAIN OBJECTIVE:

- Create comprehensive performance visualization for Air Canvas gesture recognition model

MAIN COMPONENTS:s

1. Accuracy Performance Analysis

   - Collect training accuracy data across epochs
   - Collect validation accuracy data across epochs
   - Plot accuracy curve showing model learning progression
     - X-axis: Number of Epochs
     - Y-axis: Accuracy Percentage
     - Display training and validation accuracy trends

2. Classification Performance Analysis

   - Generate Confusion Matrix
     - Rows: Actual Gesture Classes
     - Columns: Predicted Gesture Classes
     - Calculate misclassification rates
     - Visualize correct and incorrect predictions

3. Detailed Classification Metrics

   - Calculate for each gesture class:
     - Precision: Ratio of correct positive predictions
     - Recall: Proportion of actual positives correctly identified
     - F1 Score: Harmonic mean of precision and recall
   - Create bar graphs showing metrics for each gesture class

4. Model Efficiency Analysis
   - Measure inference time for each gesture
   - Create bar graph showing computational time per gesture class
   - Identify most and least computationally expensive gestures

INITIALIZATION:

1. Import required libraries
   - Matplotlib for visualization
   - NumPy for numerical operations

INPUT DATA REQUIREMENTS:

- Training accuracy values
- Validation accuracy values
- Confusion matrix
- Precision values
- Recall values
- F1 scores
- Inference times
- Gesture class names

VISUALIZATION PROCESS:

1. Create performance metrics object
2. Generate individual performance graphs
   - Accuracy curve
   - Confusion matrix
   - Precision-Recall-F1 Score
   - Inference time analysis

OUTPUT:

- Multiple visualization graphs representing model performance
- Insights into model's classification accuracy
- Understanding of computational efficiency
- Identification of potential improvement areas

ERROR HANDLING:

- Validate input data dimensions
- Handle potential visualization rendering issues
- Provide meaningful error messages

ALGORITHM: Air Canvas

Step 1: Initiate frame reading and convert the frames to the HSV color space, facilitating color detection.
Step 2: Create a canvas frame and overlay the corresponding ink buttons onto it.
Step 3: Fine-tune the track bar values to establish the mask for the colored marker.
Step 4: Preprocess the mask using morphological operations such as erosion and dilation.
Step 5: Identify contours, determine the center coordinates of the largest contour, and store them sequentially
in an array for subsequent frames. This array serves as a reference for drawing points on the canvas.
Step 6: Utilize the stored array to draw points on both the frames and the canvas.
