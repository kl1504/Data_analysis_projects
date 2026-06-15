# SNCF Delays Prediction

## Project Overview

This repository contains a comprehensive data analysis and predictive modeling project focused on understanding and predicting SNCF trains delays. By analyzing millions of rows of train data, this project identifies the root causes of delays and predict the delay time of a specific journey.

## Table of contents
- [Project Overview](#project-overview)
- [Tech Stack & Libraries](#-tech-stack--libraries)
- [Getting Started](#getting-started)
- [Prerequisites](#-prerequisites)
- [Usage](#usage)
- [Contributors](#contributors)

## 🛠️ Tech Stack & Libraries

  - Language: Python
  - Data Manipulation & Processing: pandas, numpy
  - Machine Learning: scikit-learn (Random Forest, Logistic Regression, XGBoost)
  - Data Visualization: matplotlib, seaborn, Streamlit
    
## Getting Started

## ⚙️ Prerequisites

To run the notebook, ensure you have the following installed:

- Python 3.7+
- Jupyter Notebook
- Required Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

You can install the required libraries using the following command:

   ```sh
   pip install -r requirements.txt
   ```

### Usage

1. Clone the repository:
    ```sh
    git@github.com:kl1504/Data_analysis_projects.git
    cd Data_analysis_projects
    ```
2. Launch the differents notebooks:
    ```sh
    jupyter notebook sncf_eda.ipynb
    jupyter notebook sncf_model.ipynb
    ```
4. Run the cells sequentially to generate the visualizations, analysis, and machine learning results.

5. Visualize dashboard to predict delays:
    ```sh
    streamlit run sncf_dashboard.py
    ```
## Contributors

- [Kael AVANDE](https://github.com/kl1504)

## License

This project is licensed under the [MIT](https://mit-license.org) License
