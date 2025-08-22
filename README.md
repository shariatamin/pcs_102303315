# Fake News Detection Using Graph-Based Machine Learning

### Project Description

This project develops a fake news detection system based on headline text, comparing the effectiveness of traditional machine learning models (Logistic Regression, Random Forest), deep learning models (CNN, LSTM), and a more advanced Graph Neural Network (GNN) approach. The primary focus is on the GNN to demonstrate how relationships and structure between news articles can enhance accuracy in detecting fake news, especially where traditional models fall short.

### Repository Structure

This repository contains the core files for the project:

- `102303315_PCS.ipynb`: The main Jupyter Notebook file containing all the project code, from data preprocessing to model training and evaluation.
- `102303315_PCS.csv`: The dataset used in this project.
- `README.md`: This file, which provides an overview of the project and instructions for running the code.

### Dataset

The project utilizes the "Fake and Real News Dataset" from Kaggle, which includes over 45,000 news headlines. This dataset is essential for running the code and should be included in the repository alongside other files.

### Methodology

The project employs a comprehensive approach, which includes the following stages:

1.  **Data Preprocessing:** Data has been standardized (e.g., converted to lowercase, removal of stopwords).
2.  **Feature Engineering:** TF-IDF is used for traditional models, and Word Embeddings are used for deep learning models.
3.  **Graph Construction:** News articles are treated as nodes, and connections (edges) are established based on semantic similarity and publication metadata.
4.  **Model Implementation:**
    - **Baselines:** Logistic Regression and Random Forest.
    - **Deep Learning:** CNN and LSTM.
    - **Primary Model:** Graph Convolutional Network (GCN).
5.  **Evaluation:** Model performance is assessed using metrics such as Accuracy, Precision, Recall, and F1-score.

### How to Run

To run this project, follow these steps:

1.  **Prerequisites:** Ensure you have Python and pip installed on your system.
2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/](https://github.com/shariatamin/pcs_102303315.git)
    cd pcs_102303315
    ```
3.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn torch torch-geometric
    ```
    *Note: `torch` and `torch-geometric` are required for GNN models. You may need to follow specific installation instructions for your operating system.*
4.  **Run the Notebook:** Open the `102303315_PCS.ipynb` file using Jupyter Notebook or Jupyter Lab and run all the cells.
    ```bash
    jupyter notebook
    ```

### Results and Discussion

- The GCN model, leveraging graph relationships, demonstrated improved performance in detecting fake news, particularly in cases where traditional models struggled.
- The LSTM model achieved a better F1-score in understanding textual context compared to the CNN.
- The CNN model showed excellent computational efficiency and training speed, making it suitable for real-time applications.
