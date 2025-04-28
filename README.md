# Hotel Review Sentiment Analysis Project

## 📌 Overview
This project analyzes hotel review data to predict customer satisfaction scores (positive/negative) using NLP and machine learning. The workflow includes exploratory data analysis, text preprocessing, feature engineering, and model optimization.


## 🔍 Key Insights from EDA
1. **Class Imbalance**: Positive reviews (1) outnumber negative ones (0), requiring class balancing techniques
2. **Reviewer Demographics**: Strong skew toward UK reviewers and "Other" nationalities
3. **Score Distribution**: Average scores clustered in higher ranges (7-9/10)
4. **Multicollinearity**: Location features (longitude/latitude) showed high correlation

## 🛠️ Technical Implementation
### Data Preprocessing
- **Text Cleaning**: Lowercasing, stopword removal, and custom tokenization
- **Feature Engineering**:
  - NLP: `CountVectorizer` with max 500 features (pos_/neg_ prefixes)
  - Dimensionality Reduction: PCA tested (20 components)
- **Handling Multicollinearity**: Removed longitude/latitude and redundant scoring features

### Modeling Approaches
1. **Logistic Regression**:
   - Baseline accuracy: 78.94%
   - Top predictive words:
     - Positive: 'excellent', 'comfortable', 'friendly'
     - Negative: 'dirty', 'uncomfortable', 'noisy'

2. **Decision Tree with PCA Pipeline**:
   - 5-fold CV optimization: max_depth=5, min_samples_leaf=50
   - Final accuracy: 76.12% (logistic regression remained superior)

## 📈 Performance Metrics
| Model                | Accuracy | F1-Score | ROC AUC | PR AUC  |
|----------------------|----------|----------|---------|---------|
| Logistic Regression  | 78.95%   | 0.8205   | 0.78    | 0.89    |
| Decision Tree (PCA)  | 76.12%   | 0.7812   | 0.71    | 0.83    |

## 🚀 How to Reproduce
1. Install dependencies from the `requirements.txt` file
2. Run the Jupyter notebook

## Future Directions
- Explore transformer-based NLP models (e.g., BERT) for richer text representations.
- Incorporate external data such as hotel popularity or seasonality.
- Develop a web app for real-time sentiment prediction.

---

Feel free to open issues or submit pull requests to improve the project!
