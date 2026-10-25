# ML Assignment 3: Applying, Evaluating and Comparing Multiple ML Algorithms

Muhammad Muneeb Shahzad (FA24-BSE-085)

Machine Learning Fundamentals (AIC354), COMSATS University Islamabad, Lahore Campus

## Task 2: Airbnb Price Prediction System

Live App: https://airbnb-price-fa24-bse-085.streamlit.app

Predicts the price of one night at an Airbnb listing in Pakistan (in US dollars). 5 algorithms that were not used in the Car Price system (Ridge Regression, SVR, Extra Trees, AdaBoost and Neural Network) are compared on the full dataset (1,118 listings) using 10-fold cross validation. The Neural Network had the lowest RMSE ($15.03), so it is the deployed model.

Folder: `task2-airbnb-price`

| File | What it is |
|:--|:--|
| `Airbnb Price Prediction - FA24-BSE-085.ipynb` | Notebook with all the steps |
| `Airbnb Price Prediction - FA24-BSE-085.html` and `.pdf` | Saved copies of the notebook |
| `airbnb_listings.xlsx` | Full dataset (Airbnb listings of Lahore, Islamabad, Karachi and Rawalpindi) |
| `airbnb_price_model.pkl` | Trained model |
| `app.py` | Streamlit web app |
| `requirements.txt` | Libraries needed |

## Task 1: Car Price Prediction System

The Car Price code given by the instructor. I ran both notebooks (100 instances with 3-fold cross validation, and the full dataset with 10-fold cross validation).

Folder: `task1-car-price`

## How to Run

Upload a notebook to Google Colab, click Runtime > Run all and upload the dataset when asked.

To run the web app on your own computer, open a terminal inside `task2-airbnb-price`:

```
pip install -r requirements.txt
streamlit run app.py
```
