# Detecting Demographic Bias in Online Text with Language Models project with Meta 

A machine learning project developed in collaboration with Meta mentors as part of the Break Through Tech AI Studio Fellowship (Fall 2025). This project builds language models to automatically identify demographic bias in online text, helping teams detect harmful patterns early and create safer, more equitable AI systems.

## Project Overview

This project applies fine-tuned transformer models (BERT and RoBERTa) to detect demographic bias in text datasets. By identifying biased patterns early in the data pipeline, we support the development of fairer and more responsible AI systems that can serve billions of users safely and equitably.

**Challenge Type:** Supervised Binary Classification (NLP)  
**Primary Models:** BERT (baseline), RoBERTa (optimized)  
**Dataset:** RedditBias + CrowS-Pairs (external validation)  
**Techniques:** Transfer Learning, Hyperparameter Tuning, Pseudo-Labeling, Cross-Dataset Evaluation

## Business Impact

By finding demographic bias in datasets before models are trained, this work helps prevent downstream harms including:
- Unfair content moderation
- Inequitable ranking and recommendation systems
- Misclassification of user content
- Discriminatory AI behavior affecting underrepresented groups

## Project Goals

1. **Build Automated Bias Detection**: Train transformer-based models to automatically identify biased content across multiple demographic categories
2. **Achieve High Accuracy**: Develop models with strong performance in bias detection across gender, race, orientation, and religion
3. **Create Reusable Pipeline**: Establish a reproducible workflow for evaluating dataset fairness
4. **Enable Practical Applications**: Provide tools for dataset curation and model improvement in production systems

## Datasets

### Primary Dataset: RedditBias
Source: [umanlp/RedditBias on GitHub](https://github.com/umanlp/RedditBias)

Five annotated Reddit comment datasets covering different types of demographic bias:
- **Gender**: 3,000 samples (female-related bias)
- **Orientation**: 7,993 samples (LGBTQ-related bias)
- **Race**: 3,000 samples (Black-related bias)
- **Religion 1**: 3,554 samples (Jewish-related bias)
- **Religion 2**: 10,584 samples (Muslim-related bias)

**Total**: 28,131 labeled Reddit comments with sentence-level bias annotations (binary: biased/not biased)

### External Validation: CrowS-Pairs
A standard benchmark dataset for evaluating social bias in language models, used to test cross-dataset generalization and validate that our models learned conceptual understanding of bias rather than memorizing training patterns.

**Key Challenge**: CrowS-Pairs spans multiple racial groups while our Reddit dataset focuses on specific demographic categories, providing a rigorous test of model robustness.

## Methodology

**Improvements**:
1. **Hyperparameter Optimization**:
   - Lower learning rate: 1e-5 (more stable training)
   - Higher weight decay: 0.05 (better regularization)
   - Expanded evaluation metrics: Accuracy, F1, Precision, Recall

2. **Semi-Supervised Pseudo-Labeling**:
   - Predicted labels on unlabeled Reddit data
   - Filtered predictions using 0.95 confidence threshold
   - Added high-confidence pseudo-labeled samples to training set
   - Retrained model on expanded dataset
   - **Result**: Exposed model to more diverse training examples

3. **RoBERTa Advantages**:
   - Dynamic masking: generates different masked versions across training epochs
   - Richer contextual representations
   - Particularly effective for race-related bias detection (+11.4% over BERT)

## Technical Architecture

### Models Used

**BERT (Bidirectional Encoder Representations from Transformers)**
- Pre-trained: `bert-base-uncased`
- 110M parameters
- 12 transformer layers
- Baseline for bias detection

**RoBERTa (Robustly Optimized BERT Pretraining Approach)**
- Pre-trained: `roberta-base`
- 125M parameters
- 12 transformer layers
- Improved training procedure with dynamic masking
- Final production model

### Pipeline Components

1. **Data Loading & Preprocessing**
   - Text cleaning and normalization
   - Label encoding (biased/not biased → 1/0)
   - Tokenization with model-specific tokenizers
   - Stratified splitting maintaining bias type distribution

2. **Model Training**
   - Transfer learning from pre-trained checkpoints
   - Fine-tuning on bias detection task
   - GPU-accelerated training with HuggingFace Trainer
   - Checkpointing and early stopping

3. **Evaluation**
   - Multi-metric assessment (Accuracy, Precision, Recall, F1)
   - Category-level performance analysis
   - Cross-dataset validation (CrowS-Pairs)
   - Confusion matrix and error analysis

4. **Enhancement**
   - Optuna-based hyperparameter tuning
   - Pseudo-labeling for semi-supervised learning
   - Performance visualization and analysis

## Key Findings

### 1. RoBERTa Outperforms BERT
**Finding**: RoBERTa achieved higher accuracy across all bias categories, with the largest improvement in race-related bias detection (+11.4%).

**Explanation**: RoBERTa's dynamic masking approach generates different masked versions of sentences across training epochs, learning richer contextual representations that better capture nuanced language patterns in bias detection.

### 2. Training Data Gaps Drive Performance Differences
**Finding**: Category-level analysis revealed clear correlation between training data balance and model performance:
- **Strong Performance**: Religion (94% accuracy) and Orientation (92% accuracy) - most represented in training
- **Weak Performance**: Race and Gender - limited training examples

**Implication**: Model performance directly reflects training data distribution, pointing to specific data collection needs.

## Technology Stack

**Core Libraries**:
- `transformers` (HuggingFace) - BERT and RoBERTa models
- `datasets` (HuggingFace) - Dataset loading and processing
- `torch` (PyTorch) - Deep learning framework
- `optuna` - Hyperparameter optimization

**Data Processing**:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Train/test splitting, metrics

**Visualization**:
- `matplotlib` - Basic plotting
- `seaborn` - Statistical visualizations
- Custom embedding visualizations
- Confusion matrix heatmaps

**Development Tools**:
- Google Colab - GPU-enabled notebooks
- GitHub - Version control
- Google Drive - Data storage
- Slack - Team communication
- Linear - Project management

## Results Summary

### Model Performance Comparison

| Metric | BERT (Baseline) | RoBERTa (Optimized) |
|--------|----------------|---------------------|
| Overall Accuracy | 89.3% | 93.7% |
| F1 Score | 0.88 | 0.93 |
| Precision | 0.87 | 0.92 |
| Recall | 0.89 | 0.94 |

### Category-Level Performance (RoBERTa on CrowS-Pairs)

| Bias Category | Accuracy | Notes |
|--------------|----------|-------|
| Religion | 94% | Highest representation in training |
| Orientation | 92% | Second-highest representation |
| Gender | 87% | Moderate representation |
| Race | 82% | Lowest representation, highest improvement from BERT |

### Key Achievements
✅ Successfully detected bias across 5 demographic categories  
✅ Achieved 93.7% accuracy with optimized RoBERTa model  
✅ Demonstrated cross-dataset generalization to CrowS-Pairs  
✅ Identified specific data gaps guiding future improvements  
✅ Built reusable pipeline for bias detection in text data  

## Limitations & Challenges

1. **Training Data Imbalance**: Race and religion categories have 92-94% biased examples, limiting model's ability to learn non-biased patterns
2. **Limited Demographic Coverage**: Reddit datasets focus on specific demographic groups (e.g., one racial group), limiting generalization
3. **Domain Dependence**: Models trained on Reddit may not generalize perfectly to other platforms with different discourse patterns
4. **Binary Classification**: Current approach treats bias as binary (present/absent), missing nuance and severity gradations
5. **Annotation Subjectivity**: Bias annotations reflect annotator perspectives, which may not capture all forms of bias

## Future Directions

### Data Enhancement
- **Expand Annotated Samples**: Collect more balanced examples for religion and race categories
- **Multi-Platform Data**: Include data from moderated forums, news comments, and social media beyond Reddit
- **Synthetic Data Generation**: Use language models to generate non-biased examples for underrepresented categories
- **Data Augmentation**: Apply paraphrasing and back-translation to increase training diversity

### Model Improvements
- **Lower Confidence Threshold**: Reduce pseudo-labeling threshold to capture more nuanced language patterns
- **Larger Models**: Test RoBERTa-large and other state-of-the-art transformers
- **Efficient Alternatives**: Explore distilled models (DistilBERT, DistilRoBERTa) for production deployment
- **Multi-Task Learning**: Train on multiple related tasks simultaneously (bias detection + toxicity + sentiment)

### Advanced Techniques
- **Domain Adaptation**: Techniques to improve cross-platform generalization
- **Fine-Grained Classification**: Move beyond binary to capture bias severity and type
- **Interpretability**: Add attention visualization and feature importance analysis
- **Active Learning**: Prioritize labeling examples where model is most uncertain

### Evaluation & Analysis
- **Deeper Error Analysis**: Investigate false positives and false negatives by category
- **Fairness Metrics**: Ensure model performs equitably across all demographic groups
- **Longitudinal Testing**: Track performance over time as language evolves
- **User Studies**: Validate model predictions with human judges from diverse backgrounds


## Getting Started

### Prerequisites
```bash
pip install transformers datasets torch scikit-learn pandas numpy matplotlib seaborn optuna
```

### Running the Notebook

1. Clone the repository:
```bash
git clone https://github.com/yourusername/meta-bias-detection.git
cd meta-bias-detection
```

2. Clone the RedditBias dataset:
```bash
git clone https://github.com/umanlp/RedditBias.git
```

3. Open the notebook in Google Colab or Jupyter
4. Follow the milestones sequentially:
   - Milestone 1: Data loading and preparation
   - Milestone 2: BERT baseline training
   - Milestone 3: CrowS-Pairs evaluation
   - Milestone 4: RoBERTa optimization

## What We Learned

### Technical Skills
- Fine-tuning large language models (BERT, RoBERTa) for custom classification tasks
- Implementing semi-supervised learning with pseudo-labeling
- Cross-dataset evaluation and domain adaptation
- Hyperparameter optimization with Optuna
- Using HuggingFace ecosystem for production NLP

### ML Project Skills
- Working with imbalanced datasets and stratified sampling
- Designing evaluation strategies for fairness-critical applications
- Collaborating with industry mentors on real-world problems
- Presenting technical work to both technical and non-technical audiences
- Managing ML projects with version control and project management tools

### Domain Knowledge
- Understanding different types of demographic bias in language
- Challenges in annotating subjective concepts like bias
- Importance of balanced, representative training data
- Ethical considerations in bias detection systems
- Real-world constraints and trade-offs in production ML

## Project Presentation

View our final presentation: [Meta 1D - AI Studio Final Presentation](https://docs.google.com/presentation/d/1bQfVgrj3qNP-T29msXKq9smB_6OKpdPeEsPizwm2zBU/edit?slide=id.g391364a5d05_0_20)

## Acknowledgments

This project was completed as part of the Break Through Tech AI Studio Fellowship, Fall 2025. We are grateful to:
- **Meta** for providing the challenge, mentorship, and resources
- **Break Through Tech** for the fellowship opportunity and support
- **Challenge Advisors** Candace Ross and Megan Ung for technical guidance
- **AI Studio Coach** Rajshri Jain for project management support
- The creators of the **RedditBias** and **CrowS-Pairs** datasets

## Citations

**RedditBias Dataset**:
```
Pramanick, S., Dimitrov, D., Mukherjee, R., Sharma, S., Akhtar, M. S., Nakov, P., & Chakraborty, T. (2021). 
Detecting harmful memes and their targets. arXiv preprint arXiv:2110.00413.
```

**CrowS-Pairs Dataset**:
```
Nangia, N., Vania, C., Bhalerao, R., & Bowman, S. R. (2020). 
CrowS-pairs: A challenge dataset for measuring social biases in masked language models. 
arXiv preprint arXiv:2010.00133.
```

**Break Through Tech AI Studio** | **Fall 2025** | **Meta Challenge**
