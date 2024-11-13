# Dynamic Budget Optimization Using Recurrent Neural Networks for Personalized Financial Management

**Deonna Owens** - deonnao@stanford.edu
**Ramon Silva** - ramonsgs@stanford.edu
**Project Category:**  Finance

## Problem Statement

We propose a solution to the challenge of creating personalized, dynamic budgets that adapt in real-time to users' financial behaviors and goals. Traditional budgeting tools are inherently static, offering little flexibility to account for the ever-changing nature of personal finances. This often leads to budgeting failures, with users either overspending or not adhering to their financial plans. Our project addresses this by developing a machine learning model capable of analyzing user transaction data and dynamically adjusting budgets based on real-time spending patterns and financial shifts.

## Challenges

This project presents several challenges. First, personal financial behaviors are highly variable, making it difficult to create a universal model that applies to all users while still delivering personalized recommendations, and even for the same user through time, as users' spending varies with life circumstances. Secondly,  the challenge of creating dynamic, real-time budget adjustments, which introduces both computational and algorithmic complexity, as the model must continuously process and react to new financial data.

## Dataset

Due to the difficulty in finding real-world datasets with the necessary financial variables, we will  create a simulated dataset  that mimics real user financial behavior. This dataset will be constructed as follows:
  
1. User Profiles : We will generate random data for user demographics (age, income level, occupation, family size) using statistical distributions.
2. Transaction Data : For each user, we will simulate a history of financial transactions across key categories.
3. Budget and Income : Each user will have an assigned income and a set of budget allocations for spending categories. We will simulate budget adherence and deviations.
4. Dynamic Events : We will introduce financial shocks, such as unexpected medical bills or car repairs, to simulate real-life unpredictability in finances.

## Methodology

We propose to use  Recurrent Neural Networks (RNNs) alongside LSTM (Long Short-Term Memory) to model and predict user spending patterns based on the simulated transaction data. These models can take sequential inputs (monthly or weekly transactions) to predict future expenditures. Based on these predictions, the model will recommend adjustments to the user's budget.

We will start with existing implementations of RNNs for sequential data modeling, specifically tailored to the financial domain. To improve on these methods, we plan to Introduce fine-tuned hyperparameters specific to budget modeling, Incorporate reinforcement learning (RL) to allow the model to adjust user budgets dynamically based on new financial inputs, and Integrate feedback loops, where the model learns from user behavior (e.g., when users stick to or deviate from their budget) and adjust accordingly. ​​We will also explore other implementations and related literature used for financial time-series forecasting: [1], [2], [3], [4], and [5].

## Evaluation

### Qualitative Evaluation :
Comparison Graphs  showing predicted versus actual spending over time.
Heatmaps  illustrating budget adherence across categories (e.g., food, travel, savings) for different users and personas.

### Quantitative Evaluation : 
We will use metrics such as  Mean Squared Error (MSE)  and  Mean Absolute Percentage Error (MAPE)  to evaluate the accuracy of our spending predictions.
Additionally, we will perform statistical tests (e.g., t-tests) to compare the performance of our model against a baseline static budget tool.

## References

- [1] Fjellström, C. (2022, December). Long short-term memory neural network for financial time series. In 2022 IEEE International Conference on Big Data (Big Data) (pp. 3496-3504). IEEE.
- [2] Cao, J., Li, Z., & Li, J. (2019). Financial time series forecasting model based on CEEMDAN and LSTM. Physica A: Statistical mechanics and its applications, 519, 127-139.
- [3] Dingli, A., & Fournier, K. S. (2017). Financial time series forecasting-a deep learning approach. International Journal of Machine Learning and Computing, 7(5), 118-122.
- [4] Sako, K., Mpinda, B. N., & Rodrigues, P. C. (2022). Neural networks for financial time series forecasting. Entropy, 24(5), 657.
- [5] Zhao, Y., Zhang, W., & Liu, X. (2024). Grid search with a weighted error function: Hyper-parameter optimization for financial time series forecasting. Applied Soft Computing, 154, 111362.
