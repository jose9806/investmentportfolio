import json
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import os
import boto3
import io
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro, kurtosis

s3 = boto3.client('s3') 

# Function to fetch and preprocess stock data
def fetch_data(tickers, start_date, end_date):
    # Fetch stock data
    data = yf.download(tickers, start=start_date, end=end_date)
    adjusted_close = data['Adj Close'].copy()
    adjusted_close.ffill(inplace=True)  # Forward fill
    adjusted_close.bfill(inplace=True)  # Backward fill

    # Save to S3
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        adjusted_close.to_excel(writer)
    excel_buffer.seek(0)
    s3.put_object(Bucket='your-bucket-name', Key='historical_stock_data.xlsx', Body=excel_buffer.read())

    # Calculate raw returns
    return adjusted_close.pct_change().dropna()

def assess_normality(returns):
    p_values = returns.apply(shapiro).apply(lambda x: x[1])  # Shapiro-Wilk test
    excess_kurtosis = kurtosis(returns, fisher=False) - 3  # Excess Kurtosis
    is_normal = (p_values > 0.05) & (abs(excess_kurtosis) < 1)  # Threshold for normality
    return is_normal.all()

def choose_distribution_model(returns):
    if assess_normality(returns):
        return 'normal'
    else:
        return 't-distribution'

# Portfolio Optimization Function
def optimize_portfolio(risk_tolerance, returns, risk_free_rate):
    distribution_model = choose_distribution_model(returns)

    if distribution_model == 'normal':
        # Use existing functions (assuming they are based on normal distribution)
        return_calc = portfolio_return
        vol_calc = portfolio_volatility
        sharpe_calc = lambda w: negative_sharpe_ratio(w, returns, returns.cov(), risk_free_rate)
    else:
        # Use T-Distribution based functions
        return_calc = portfolio_return_t_dist
        vol_calc = portfolio_volatility_t_dist
        sharpe_calc = lambda w: negative_sharpe_ratio_t_dist(w, returns, risk_free_rate)

    num_assets = len(returns.columns)
    initial_guess = np.random.random(num_assets)
    initial_guess /= np.sum(initial_guess)
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

    if risk_tolerance == 'aggressive':
        objective_function = lambda x: -return_calc(x, returns)
    elif risk_tolerance == 'moderate':
        objective_function = sharpe_calc
    else:  # Conservative
        objective_function = lambda x: vol_calc(x, returns)

    result = minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        raise ValueError(f"Optimization did not converge: {result.message}")

    return result.x

# Functions for portfolio metrics
def portfolio_return(weights, returns):
    return np.sum(weights * returns.mean()) * 252

def portfolio_volatility(weights, covariance_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights))) * np.sqrt(252)

def negative_sharpe_ratio(weights, returns, covariance_matrix, risk_free_rate):
    p_ret = portfolio_return(weights, returns)
    p_vol = portfolio_volatility(weights, covariance_matrix)
    return -(p_ret - risk_free_rate) / p_vol

def check_sum(weights):
    return np.sum(weights) - 1

def minimize_volatility(weights, covariance_matrix):
    return portfolio_volatility(weights, covariance_matrix)

def portfolio_return_t_dist(weights, returns):
    mean_returns = returns.mean()
    return np.sum(weights * mean_returns) * 252

def portfolio_volatility_t_dist(weights, returns):
    df, loc, scale = stats.t.fit(returns)
    cov_matrix = returns.cov() * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return volatility * np.sqrt((df - 2) / df)

def negative_sharpe_ratio_t_dist(weights, returns, risk_free_rate):
    p_ret = portfolio_return_t_dist(weights, returns)
    p_vol = portfolio_volatility_t_dist(weights, returns)
    return -(p_ret - risk_free_rate) / p_vol

# Functions for efficient frontier
def efficient_frontier(returns, covariance_matrix, num_portfolios=100):
    num_assets = len(returns.columns)
    bounds = tuple((0, 1) for asset in range(num_assets))
    # Adjust the range of target returns
    lower_bound = np.percentile(returns.mean(), 10) * 252
    upper_bound = np.percentile(returns.mean(), 90) * 252
    portfolio_means = np.linspace(lower_bound, upper_bound, num_portfolios)
    efficient_portfolios = []
    
    for target_mean in portfolio_means:
        constraints = [{'type': 'eq', 'fun': check_sum},
                       {'type': 'eq', 'fun': lambda x: portfolio_return(x, returns) - target_mean}]
        
        result = minimize(portfolio_volatility, num_assets * [1. / num_assets], args=(covariance_matrix,),
                          method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            efficient_portfolios.append(result.x)
    
    return np.array(efficient_portfolios)

# Monte Carlo Simulation Function
def monte_carlo_simulation(returns, weights, num_simulations, day_counts):
    results = {}
    for num_days in day_counts:
        portfolio_ret = np.sum(returns.mean() * weights) * num_days
        portfolio_std_dev = portfolio_volatility(weights, returns.cov()) * np.sqrt(num_days)
        simulated_end_values = np.zeros(num_simulations)
        for i in range(num_simulations):
            simulated_returns = np.random.normal(portfolio_ret, portfolio_std_dev, 1)
            simulated_end_values[i] = simulated_returns
        results[num_days] = {
            "mean": np.mean(simulated_end_values),
            "median": np.median(simulated_end_values),
            "std_dev": np.std(simulated_end_values),
            "percentile_5": np.percentile(simulated_end_values, 5),
            "percentile_95": np.percentile(simulated_end_values, 95),
            "var_95": np.percentile(simulated_end_values, 5),
            "cvar_95": simulated_end_values[simulated_end_values <= np.percentile(simulated_end_values, 5)].mean()
        }
    return results

# Function to calculate expected returns for different time horizons
def calculate_optimization_expected_returns(returns, weights, day_counts):
    annualized_returns = returns.mean() * 252
    expected_returns = {}
    for num_days in day_counts:
        expected_return = np.dot(annualized_returns, weights) * (num_days / 252)
        expected_returns[num_days] = expected_return
    return expected_returns

def save_plot_to_s3(fig, bucket, filename):
    img_data = io.BytesIO()
    fig.savefig(img_data, format='png')
    img_data.seek(0)
    s3.put_object(Bucket=bucket, Key=filename, Body=img_data, ContentType='image/png')

def plot_efficient_frontier(returns, cov_matrix, optimal_weights, tickers, efficient_portfolios):
    plt.figure(figsize=(12, 8))
    plt.title('Efficient Frontier with Optimal Portfolio')
    plt.xlabel('Volatility (Standard Deviation)')
    plt.ylabel('Expected Return')

    portfolio_risks = [portfolio_volatility(weights, cov_matrix) for weights in efficient_portfolios]
    portfolio_returns = [portfolio_return(weights, returns) for weights in efficient_portfolios]

    plt.plot(portfolio_risks, portfolio_returns, 'o-', markersize=2, label='Efficient Frontier')
    optimal_portfolio_volatility = portfolio_volatility(optimal_weights, cov_matrix)
    optimal_portfolio_return = portfolio_return(optimal_weights, returns)
    plt.scatter(optimal_portfolio_volatility, optimal_portfolio_return, color='red', marker='*', s=100, label='My Optimal Portfolio')

    plt.grid(True)
    plt.legend()

    # Create a Figure object and save to S3
    fig = plt.gcf()  # Get current figure
    save_plot_to_s3(fig, 'your-bucket-name', 'efficient_frontier.png')    

def lambda_handler(event, context):
    try:
        # Extract parameters from the event object (e.g., API Gateway)
        tickers = os.environ.get("tickers", "AAPL,MSFT,GOOG").split(',')
        start_date = os.environ.get("start_date", "2010-01-01")
        end_date = os.environ.get("end_date", "2024-01-01")
        risk_tolerance = os.environ.get("risk_tolerance", "moderate")
        risk_free_rate = os.environ.get("risk_free_rate", 0.03)
        time_horizons = os.environ.get("time_horizons", [252, 756, 1260])
        num_simulations = os.environ.get("num_simulations", 10000)

        # Fetch data
        returns = fetch_data(tickers, start_date, end_date)
        cov_matrix = returns.cov()

        # Optimization
        optimal_weights = optimize_portfolio(risk_tolerance, returns, cov_matrix, risk_free_rate)
        print("Optimal Portfolio Weights:")
        for ticker, weight in zip(tickers, optimal_weights):
            if weight > 1e-3:
                print(f"{ticker}: {weight:.2%}")            

        # Efficient Frontier Calculation
        efficient_portfolios = efficient_frontier(returns, cov_matrix)

        # Plotting the Efficient Frontier with the Optimal Portfolio
        plot_efficient_frontier(returns, cov_matrix, optimal_weights, tickers, efficient_portfolios)
        
        # Monte Carlo Simulation
        monte_carlo_results = monte_carlo_simulation(returns, optimal_weights, num_simulations, time_horizons)
        print("\nMonte Carlo Simulation Results:")
        for days, result in monte_carlo_results.items():
            print(f"\nResults for {days//252} year(s):")
            print(f"Mean Return: {result['mean']:.2%}")
            print(f"Median Return: {result['median']:.2%}")
            print(f"Standard Deviation: {result['std_dev']:.2%}")
            print(f"5th Percentile (VaR 95%): {result['percentile_5']:.2%}")
            print(f"95th Percentile: {result['percentile_95']:.2%}")
            print(f"Conditional Value at Risk (CVaR 95%): {result['cvar_95']:.2%}")

        # Optimization-Based Expected Returns
        optimization_returns = calculate_optimization_expected_returns(returns, optimal_weights, time_horizons)
        print("\nOptimization-Based Expected Returns:")
        for days, expected_return in optimization_returns.items():
            print(f"For {days//252} year(s): {expected_return:.2%}")  

    # Return a response
        return {
            "statusCode": 200,
            "body": json.dumps("Portfolio analysis completed successfully!")
        }        

    except Exception as e:
        # Return an error response
        return {
            "statusCode": 500,
            "body": json.dumps(f"An error occurred: {str(e)}")
        }
    
lambda_handler({},{})