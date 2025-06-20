import pandas as pd
import joblib
import numpy as np
from data_load import data_load
from sklearn.model_selection import train_test_split


def split_conformal_test(X_pooled,y_pooled, model,alpha = 0.1):
    portion_within_list = []
    for i in range(1,101):
        X_test, X_calib, y_test, y_calib = train_test_split(X_pooled, y_pooled, test_size=0.5, random_state=i)
        q_hat = full_split_conformal_q(X_calib,y_calib,model=model,alpha=alpha)
        y_test_pred = model.predict(X_test)
        within_bounds = np.logical_and(
        y_test>=y_test_pred - q_hat,
        y_test<=y_test_pred + q_hat,
        )
        portion_within = np.mean(within_bounds)
        portion_within_list.append(portion_within)
    average_portion = np.average(portion_within_list)
    print(f"Portion of y_test within ±q_hat of prediction on average is: {average_portion:.2%}")

def full_split_conformal_q(X_pooled,y_pooled, model,alpha = 0.1):
    y_pooled_pred = model.predict(X_pooled)
    residuals = np.abs(y_pooled - y_pooled_pred)
    n_cal = len(y_pooled)
    q_hat = np.quantile(residuals, (1 - alpha) * (1+1/n_cal))
    return q_hat

def main():

    model = joblib.load("model_dataset/xgboost_model18june.pkl")
    # Generating the q_hat
    X_train,X_test,y_train,y_test = data_load()

    split_conformal_test(X_test,y_test, model, 0.1)
    q_hat = full_split_conformal_q(X_test,y_test, model,0.1)
    print(f"The q_hat from the total calibration set was {q_hat}")

    pd.DataFrame({'q_hat': [q_hat]}).to_csv('q_hat.csv', index=False)

# mm okay idea
# Give pooled data to the tester and make it do many different splits for testing and return the average or median


if __name__ == "__main__":
    main()