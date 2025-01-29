# Student Performance Prediction

## Overview
The **Student Performance Prediction** project aims to analyze various factors affecting student performance and build predictive models to estimate students' academic success. By leveraging machine learning techniques, the project seeks to identify key indicators that influence student grades and provide actionable insights for educators and students.

## Features
- Data preprocessing and feature engineering
- Exploratory Data Analysis (EDA)
- Machine Learning model training and evaluation
- Performance comparison of different models
- Visualizations and insights generation
- Deployment using Streamlit for interactive predictions

## Installation
To set up the project on your local machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Prajeesh-A/Student_Performance_Prediction.git
   cd Student_Performance_Prediction
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
This project requires a dataset containing student performance metrics. You can use publicly available datasets such as:
- UCI Student Performance Dataset
- Kaggle datasets related to student scores

Ensure the dataset includes relevant features such as attendance, study hours, past grades, socioeconomic factors, etc.

## Usage
1. Load and preprocess the dataset using:
   ```python
   python Student_Performance_EDA.ipynb
   ```
2. Train models:
   ```python
   python train.py
   ```
3. Evaluate model performance:
   ```python
   python evaluate.py
   ```
4. Visualize insights:
   ```python
   python visualize.py
   ```

## Deployment with Streamlit
To deploy the Streamlit application:
1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run student.py
   ```
3. Open the displayed URL in your browser to interact with the app.

## Models Used
- Linear Regression
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- Neural Networks

## Results & Insights
- Performance metrics (Accuracy, RMSE, R² Score)
- Feature importance analysis
- Student performance trend visualization

## File Structure
```
Student_Performance_Prediction/
│-- LICENSE
│-- README.md
│-- requirements.txt
│-- StudentPerformanceFactors.csv
│-- features.csv
│-- model.pkl
│-- student.py
│-- Student_Performance_EDA.ipynb
```

## Contributing
Contributions are welcome! If you'd like to contribute:
1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Added new feature"`).
4. Push the changes to your branch (`git push origin feature-branch`).
5. Submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any queries or discussions, feel free to reach out at:
- **Your Name**: Prajeesh
- **Email**: prajeep6@gmail.com
- **GitHub**: [yourgithub](https://github.com/Prajeesh-A)

---

Happy Coding! 🚀

