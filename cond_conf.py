# ------------------------------------------------------------------------------------------------------------
# Importing the packages
from clover import LocartSplit, RegressionScore
import joblib
from data_load import data_load

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------------------------------------
# The core function

def LOFOREST(X_train, X_calib, y_train, y_calib, model, alpha=0.1):
    loforest = LocartSplit(RegressionScore, model, alpha = alpha, cart_type = "RF", is_fitted = True)
    loforest.fit(X_train,y_train)
    loforest.calib(X_calib, y_calib)
    return loforest

# ------------------------------------------------------------------------------------------------------------
# loforest testing
def overal_coverage(X_test, y_test, loforest_calib):
    bounds = loforest_calib.predict(X_test)
    is_inside_bounds = (y_test >= bounds[:, 0]) & (y_test <= bounds[:, 1])
    return np.mean(is_inside_bounds)

def conditional_coverage(X_test, y_test, loforest_calib):
    # Calculating the prediction interval size for binning purposes
    bounds = loforest_calib.predict(X_test)
    prediction_interval_size = bounds[:, 1] - bounds[:, 0]
    # Binning based on prediction interval size
    df = pd.DataFrame({
        "interval_size":prediction_interval_size,
        "y_true":y_test,
        "lower_bound":bounds[:,0],
        "upper_bound":bounds[:,1]
    })
    n_bins = 10
    df['bins'], bins = pd.qcut(df['interval_size'], q = n_bins, retbins = True,labels=False)
    # Now calculating the statistics we want for each bin
    coverages = []
    for b in sorted(df['bins'].unique()):
        bin_data = df[df['bins']==b]
        covered = ((bin_data['y_true'] >= bin_data['lower_bound']) &
                   (bin_data['y_true'] <= bin_data['upper_bound']))
        coverage = covered.mean()
        coverages.append(coverage)
    return np.min(coverages)

def statistics(X_train, X_pooled, y_train, y_pooled, model, alpha = 0.1, random_seed = 42):
    # Splitting the data between calibration set and the test set
    X_calib, X_test, y_calib, y_test = train_test_split(X_pooled,y_pooled, random_state=random_seed, test_size=0.5)
    # Training the model on calibration set
    loforest_calib = LOFOREST(X_train=X_train, X_calib=X_calib, y_train=y_train, y_calib=y_calib, model = model, alpha = alpha)
    # Test the model on the test data
    ## Overall coverage
    overal_cov = overal_coverage(X_test, y_test, loforest_calib)
    ## Conditional coverage
    cond_cov = conditional_coverage(X_test, y_test, loforest_calib)
    return overal_cov,cond_cov

def robust_test(X_train, X_pooled, y_train, y_pooled, model, alpha = 0.1, iterations = 20):
    overal_cov_stats = []
    cond_cov_stats = []
    for i in range(1,iterations+1):
        overal_cov, cond_cov = statistics(X_train, X_pooled, y_train, y_pooled, model, alpha = alpha, random_seed = i)
        overal_cov_stats.append(overal_cov)
        cond_cov_stats.append(cond_cov)
        if i % 10 == 0:
            print(f"{i}/{iterations}")
    average_overal_cov = np.mean(overal_cov_stats)
    average_cond_cov = np.mean(cond_cov_stats)
    return average_overal_cov, average_cond_cov

# ----------------------------------------------------------------------------------------------------------------
# loforest graphs
## Histogram of prediction intervals
def histogram(values, name):
    plt.hist(values, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {name}')
    plt.savefig(f'hist_of_{name}.png', dpi=300, bbox_inches='tight')
    plt.close()
def prediction_interval_histogram(X_test, loforest):
    bounds = loforest.predict(X_test)
    prediction_intervals = bounds[:,1]-bounds[:,0]
    histogram(prediction_intervals, "prediction interval size")

## Feature importances
def horizontal_bar_chart(importances):
    plt.figure(figsize=(10, 6))
    importances.plot(kind='barh', color='skyblue')
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances')
    plt.gca().invert_yaxis()  # Most important at top
    plt.tight_layout()
    plt.savefig(f'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
def top_20_important(loforest,X_test):
    importances = loforest.RF.feature_importances_
    feature_names = X_test.columns
    importance_series = pd.Series(importances, index=feature_names)
    top_20 = importance_series.sort_values(ascending=False).head(20)
    return top_20
def plot_importances(loforest,X_test):
    top_20 = top_20_important(loforest,X_test)
    horizontal_bar_chart(top_20)

### Scatter plots for top 3 features and the prediction interval
def feature_scatter(x_values, y_values,col_name):
    plt.scatter(x_values, y_values, color='blue', marker='o')

    # Add labels and title
    plt.xlabel(col_name)
    plt.ylabel('Prediction Interval Size')
    plt.title(f'Prediciton interval size plotted against {col_name}')

    # Show plot
    plt.savefig(f'scatter_pred_interval_{col_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
def top_3_scatterplot(X_test, loforest):
    bounds = loforest.predict(X_test)
    top_20 = top_20_important(loforest,X_test)
    top_3 = top_20[:3]
    y_values = bounds[:,1]-bounds[:,0]
    for col_name in top_3.index:
        x_values = X_test[col_name]
        feature_scatter(x_values, y_values,col_name)
# Test Zone
def plot_pred_intervals(y_pred, y_true, lower_bounds, upper_bounds):
    import matplotlib.colors as mcolors
    fig, ax = plt.subplots(figsize=(10,6))  # create figure and axis

    interval_lengths = upper_bounds - lower_bounds

    norm = mcolors.Normalize(vmin=interval_lengths.min(), vmax=interval_lengths.max())
    cmap = plt.cm.viridis

    # Scatter plot on ax
    ax.scatter(y_pred, y_true, color='blue', label='True vs Predicted')

    # Vertical lines colored by length
    for x, low, up, length in zip(y_pred, lower_bounds, upper_bounds, interval_lengths):
        ax.plot([x, x], [low, up], color=cmap(norm(length)), alpha=0.8)

    # Dashed diagonal line y_pred = y_true
    min_val = min(min(y_pred), min(y_true))
    max_val = max(max(y_pred), max(y_true))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='y_pred = y_true')

    # Create ScalarMappable for colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add colorbar to the figure, linked to ax
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Prediction Interval Length')

    ax.set_xlabel('Predicted values')
    ax.set_ylabel('True values')
    ax.set_title('Prediction Intervals Colored by Length')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(f'scat_plt_pred_int.png', dpi=300, bbox_inches='tight')
    plt.close()

def interval_scatter_plot(X_test,y_test,model, loforest):
    sampled = X_test.sample(n=15, random_state=42)  # random_state for reproducibility
    sampled_indices = sampled.index
    sampled_y = y_test.loc[sampled_indices]

    y_pred = model.predict(sampled)
    bounds = loforest.predict(sampled)
    lower_bounds = bounds[:,0]
    upper_bounds = bounds[:,1]
    plot_pred_intervals(y_pred,sampled_y,lower_bounds,upper_bounds)

# ---------------------------------------------------------------------------------------------------------------
# Running the script
def main():
    # Import the model
    xgboost = joblib.load("model_dataset/xgboost_model18june.pkl")
    # Import the relevant data
    X_train,X_test,y_train,y_test = data_load()
    # Calculate relevant statistics for the loforest model
    #overal_cov, cond_cov = robust_test(X_train, X_test, y_train, y_test, xgboost, alpha = 0.1, iterations=50)
    # Generate the complete loforest model
    loforest_full = LOFOREST(X_train, X_test, y_train, y_test, xgboost, alpha=0.1)
    joblib.dump(loforest_full, 'loforest_full_model.pkl')
    # Generate graphs for the loforest model
    ## Histogram of prediction intervals
    prediction_interval_histogram(X_test, loforest_full)
    ## feature importances
    plot_importances(loforest_full,X_test)
    ## scatter plots for the two most important features
    top_3_scatterplot(X_test=X_test, loforest=loforest_full)
    ## Special scatter plot
    interval_scatter_plot(X_test,y_test,xgboost, loforest_full)

if __name__ == "__main__":
    main()