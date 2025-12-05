# Protein Structure Classification with Machine Learning

This folder contains content from when I led a hands-on machine learning workshop series developed for the Biotech Club at UT Dallas. This project teaches core ML concepts through the practical application of classifying protein structures from the Protein Data Bank (PDB).

## Project Overview

This project uses protein metadata from crystallography experiments to predict protein functional classifications. Built as a series of progressive workshops, participants learn foundational machine learning techniques while working with real biological data.

**Dataset**: 99,529 protein structures from the Protein Data Bank  
**Task**: Multi-class classification predicting protein function from structural metadata  
**Model**: Random Forest classifier with hyperparameter optimization

## Key Concepts Covered

### Week 1: Exploratory Data Analysis (EDA)
- Data loading and inspection with pandas
- Handling missing values and data quality assessment
- Statistical analysis of numerical features
- Visualization techniques with matplotlib and seaborn
- Understanding class distributions and imbalances

### Week 2: Data Preprocessing & Feature Engineering
- Categorical encoding strategies (one-hot encoding)
- Feature selection and correlation analysis
- Train-test split methodology
- Data cleaning and preparation for modeling

### Week 3: Model Training & Evaluation
- Random Forest classifier implementation
- Cross-validation techniques
- Model evaluation metrics (accuracy, classification reports, confusion matrices)
- Understanding overfitting and model generalization

### Week 4: Hyperparameter Tuning
- Grid search methodology
- Parameter optimization for Random Forest
- Interpreting cross-validation results
- Model performance comparison

### Final: Prediction Interface
- Building user-friendly prediction functions
- Feature engineering for new data points
- Probability estimation and confidence scoring
- Creating interpretable model outputs

## Features Used

**Numerical Features:**
- Residue count
- Resolution (Angstroms)
- Molecular weight
- Crystallization temperature (Kelvin)
- Matthews density coefficient
- Percent solvent content
- pH value
- Publication year

**Categorical Features:**
- Experimental technique
- Macromolecule type
- Crystallization method


## Workshop Structure

Each section builds on previous concepts, designed for progressive learning:

1. **Week 1**: Understanding the data and domain
2. **Week 2**: Preparing data for machine learning
3. **Week 3**: Training and evaluating models
4. **Week 4**: Optimizing model performance
5. **Final**: Deploying predictions

## Skills Developed

- Data manipulation with pandas
- Statistical analysis and visualization
- Feature engineering techniques
- Machine learning model selection
- Hyperparameter optimization
- Model evaluation and interpretation
- Building production-ready prediction interfaces

## Dataset Information

The dataset contains metadata from protein crystallography experiments, including structural properties, experimental conditions, and functional classifications. Each protein structure is labeled with its biological function, creating a multi-class classification problem.

## Model Performance

The optimized Random Forest classifier achieves robust performance through:
- Grid search hyperparameter tuning
- 5-fold cross-validation
- Balanced handling of class imbalances
- Feature importance analysis

## Future Enhancements

- Incorporate additional PDB metadata
- Implement deep learning approaches
- Add sequence-based features
- Deploy as a web application
- Expand to regression tasks (predicting resolution, molecular weight)

## About

This project was developed as part of the Mini Missions initiative at the Biotech Club, UT Dallas. The goal is to make machine learning accessible to students through structured, hands-on learning with real biological data.
