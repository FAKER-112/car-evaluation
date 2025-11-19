# problem statement 
**Problem Statement**

The Car Evaluation Database provides a structured dataset used to predict the overall acceptability of a car based on six categorical attributes: buying price, maintenance cost, number of doors, passenger capacity, luggage boot size, and safety rating. The original data originates from a hierarchical decision model designed to evaluate cars through intermediate decision layers (PRICE, TECH, COMFORT). However, in this dataset, the hierarchical structure has been removed, leaving a direct mapping between the six input attributes and the final class label.

The predictive task is to build a classification model that accurately determines car acceptability into one of four possible classes: unacceptable, acceptable, good, or very good. No missing values exist, and all 1,728 possible attribute combinations are represented, making the attribute space complete. The class distribution is highly imbalanced, with the majority of instances labeled as unacceptable.

This dataset serves as a benchmark for machine learning methods, particularly those involving constructive induction, hierarchical model reconstruction, and classification under class imbalance. The problem requires developing and evaluating algorithms capable of learning meaningful decision boundaries within a fully discrete attribute space while handling skewed class distributions.




data source: https://archive.ics.uci.edu/dataset/19/car+evaluation