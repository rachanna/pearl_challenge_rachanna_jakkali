import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD SUBMISSION DATA
# =============================================================================

def load_submission_data(file_path):
    """Load submission data from CSV file"""
    df = pd.read_csv(file_path)
    print(f"Loaded submission data: {len(df)} predictions")
    print(f"Columns: {df.columns.tolist()}")
    
    # Standardize column names
    if 'Predicted_farmer_income_in_numeric_format' in df.columns:
        df = df.rename(columns={'Predicted_farmer_income_in_numeric_format': 'predicted_income'})
    
    return df

# =============================================================================
# GROUND TRUTH ESTIMATION METHODS
# =============================================================================

def estimate_ground_truth_method1(predicted_values, target_mape=19.18):
    """
    Method 1: Assume errors are normally distributed around the target MAPE
    This creates plausible actual values that would produce the target MAPE
    """
    print(f"\nMethod 1: Normal Error Distribution (Target MAPE: {target_mape}%)")
    
    # Generate random percentage errors with mean = target_mape
    # Use a reasonable standard deviation (half the mean)
    np.random.seed(42)  # For reproducibility
    n_samples = len(predicted_values)
    
    # Generate percentage errors (both positive and negative)
    error_std = target_mape / 2  # Standard deviation
    percentage_errors = np.random.normal(0, error_std, n_samples)
    
    # Ensure some constraint on error range (clip extreme outliers)
    percentage_errors = np.clip(percentage_errors, -50, 50)
    
    # Calculate actual values: actual = predicted / (1 + error/100)
    # If predicted is over-estimated by 20%, then actual = predicted / 1.20
    actual_values = predicted_values / (1 + percentage_errors/100)
    
    # Ensure realistic income bounds
    actual_values = np.clip(actual_values, 50000, 100000000)
    
    # Calculate achieved MAPE
    achieved_mape = mean_absolute_percentage_error(actual_values, predicted_values) * 100
    
    print(f"  Generated {n_samples} actual values")
    print(f"  Achieved MAPE: {achieved_mape:.2f}%")
    print(f"  Actual income range: ₹{actual_values.min():,.0f} - ₹{actual_values.max():,.0f}")
    
    return actual_values, achieved_mape

def estimate_ground_truth_method2(predicted_values, target_mape=19.18):
    """
    Method 2: Based on income range patterns from training data
    Use different error patterns for different income ranges
    """
    print(f"\nMethod 2: Income Range-Based Errors (Target MAPE: {target_mape}%)")
    
    np.random.seed(42)
    actual_values = predicted_values.copy()
    
    # Apply different error patterns based on predicted income ranges
    # Lower income: typically under-predicted (model conservative)
    # Higher income: typically over-predicted (model optimistic)
    
    for i, pred_income in enumerate(predicted_values):
        if pred_income < 300000:  # Low income range
            # Model tends to under-predict (actual is higher than predicted)
            error_pct = np.random.normal(target_mape, target_mape/3)
            actual_values[i] = pred_income * (1 + abs(error_pct)/100)
            
        elif pred_income < 800000:  # Mid income range  
            # More balanced errors
            error_pct = np.random.normal(0, target_mape/2)
            actual_values[i] = pred_income * (1 + error_pct/100)
            
        else:  # High income range
            # Model tends to over-predict (actual is lower than predicted)
            error_pct = np.random.normal(-target_mape/2, target_mape/3)
            actual_values[i] = pred_income * (1 + error_pct/100)
    
    # Ensure realistic bounds
    actual_values = np.clip(actual_values, 50000, 100000000)
    
    # Calculate achieved MAPE
    achieved_mape = mean_absolute_percentage_error(actual_values, predicted_values) * 100
    
    print(f"  Generated {n_samples} actual values with income-based patterns")
    print(f"  Achieved MAPE: {achieved_mape:.2f}%")
    print(f"  Actual income range: ₹{actual_values.min():,.0f} - ₹{actual_values.max():,.0f}")
    
    return actual_values, achieved_mape

def estimate_ground_truth_method3(predicted_values, target_mape=19.18):
    """
    Method 3: Iterative adjustment to exactly match target MAPE
    """
    print(f"\nMethod 3: Iterative MAPE Matching (Target MAPE: {target_mape}%)")
    
    np.random.seed(42)
    actual_values = predicted_values.copy()
    
    # Start with random perturbations
    perturbations = np.random.normal(0, 0.2, len(predicted_values))
    actual_values = predicted_values * (1 + perturbations)
    
    # Iteratively adjust to match target MAPE
    max_iterations = 100
    tolerance = 0.01
    
    for iteration in range(max_iterations):
        current_mape = mean_absolute_percentage_error(actual_values, predicted_values) * 100
        diff = target_mape - current_mape
        
        if abs(diff) < tolerance:
            break
            
        # Adjust actual values proportionally
        adjustment_factor = 1 + (diff / target_mape) * 0.1
        actual_values = actual_values * adjustment_factor
        
        # Ensure bounds
        actual_values = np.clip(actual_values, 50000, 100000000)
    
    final_mape = mean_absolute_percentage_error(actual_values, predicted_values) * 100
    
    print(f"  Converged after {iteration+1} iterations")
    print(f"  Final MAPE: {final_mape:.3f}%")
    print(f"  Actual income range: ₹{actual_values.min():,.0f} - ₹{actual_values.max():,.0f}")
    
    return actual_values, final_mape

# =============================================================================
# ANALYSIS AND COMPARISON FUNCTIONS
# =============================================================================

def analyze_predictions(farmer_ids, actual_values, predicted_values):
    """Comprehensive analysis of prediction performance"""
    
    # Calculate metrics
    mape = mean_absolute_percentage_error(actual_values, predicted_values) * 100
    mae = mean_absolute_error(actual_values, predicted_values)
    r2 = r2_score(actual_values, predicted_values)
    
    # Calculate error statistics
    absolute_errors = np.abs(actual_values - predicted_values)
    percentage_errors = np.abs((actual_values - predicted_values) / actual_values) * 100
    
    print(f"\nPERFORMACE METRICS:")
    print(f"  MAPE: {mape:.3f}%")
    print(f"  MAE: ₹{mae:,.2f}")
    print(f"  R²: {r2:.4f}")
    
    print(f"\nERROR ANALYSIS:")
    print(f"  Mean absolute error: ₹{np.mean(absolute_errors):,.2f}")
    print(f"  Median absolute error: ₹{np.median(absolute_errors):,.2f}")
    print(f"  Std of errors: ₹{np.std(absolute_errors):,.2f}")
    
    print(f"\nACCURACY THRESHOLDS:")
    for threshold in [5, 10, 15, 18, 25]:
        within_threshold = np.mean(percentage_errors <= threshold) * 100
        print(f"  Within {threshold}%: {within_threshold:.1f}% of predictions")
    
    # Create analysis DataFrame
    analysis_df = pd.DataFrame({
        'FarmerID': farmer_ids,
        'actual_income': actual_values.astype(int),
        'predicted_income': predicted_values.astype(int),
        'absolute_error': absolute_errors.astype(int),
        'percentage_error': percentage_errors
    })
    
    return analysis_df, {
        'mape': mape,
        'mae': mae,
        'r2': r2,
        'mean_abs_error': np.mean(absolute_errors),
        'median_abs_error': np.median(absolute_errors),
        'std_error': np.std(absolute_errors)
    }

def create_visualizations(analysis_df):
    """Create comprehensive visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Prediction Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Predictions vs Actual scatter plot
    axes[0, 0].scatter(analysis_df['actual_income'], analysis_df['predicted_income'], 
                      alpha=0.6, s=20, color='blue')
    min_val = min(analysis_df['actual_income'].min(), analysis_df['predicted_income'].min())
    max_val = max(analysis_df['actual_income'].max(), analysis_df['predicted_income'].max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Income (₹)')
    axes[0, 0].set_ylabel('Predicted Income (₹)')
    axes[0, 0].set_title('Predictions vs Actual')
    axes[0, 0].legend()
    axes[0, 0].ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    # 2. Residuals plot
    residuals = analysis_df['actual_income'] - analysis_df['predicted_income']
    axes[0, 1].scatter(analysis_df['predicted_income'], residuals, alpha=0.6, s=20, color='green')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Income (₹)')
    axes[0, 1].set_ylabel('Residuals (₹)')
    axes[0, 1].set_title('Residuals vs Predicted')
    axes[0, 1].ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    # 3. Error distribution
    axes[0, 2].hist(analysis_df['percentage_error'], bins=50, alpha=0.7, 
                   color='orange', edgecolor='black')
    axes[0, 2].axvline(x=18, color='red', linestyle='--', lw=2, label='18% Target')
    axes[0, 2].axvline(x=analysis_df['percentage_error'].mean(), color='blue', 
                      linestyle='-', lw=2, label=f'Mean: {analysis_df["percentage_error"].mean():.1f}%')
    axes[0, 2].set_xlabel('Absolute Percentage Error (%)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distribution of Prediction Errors')
    axes[0, 2].legend()
    
    # 4. Income range analysis
    income_ranges = [(0, 2e5, '0-2L'), (2e5, 5e5, '2-5L'), (5e5, 10e5, '5-10L'), (10e5, float('inf'), '10L+')]
    range_mapes = []
    range_labels = []
    
    for min_val, max_val, label in income_ranges:
        mask = (analysis_df['actual_income'] >= min_val) & (analysis_df['actual_income'] < max_val)
        if mask.sum() > 10:  # At least 10 samples
            range_mape = analysis_df.loc[mask, 'percentage_error'].mean()
            range_mapes.append(range_mape)
            range_labels.append(f"{label}\n(n={mask.sum()})")
    
    if range_mapes:
        colors = ['green' if mape < 18 else 'orange' if mape < 25 else 'red' for mape in range_mapes]
        bars = axes[1, 0].bar(range_labels, range_mapes, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 0].axhline(y=18, color='red', linestyle='--', lw=2, label='18% Target')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].set_title('Performance by Income Range')
        axes[1, 0].legend()
        
        # Add value labels on bars
        for bar, mape in zip(bars, range_mapes):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                           f'{mape:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 5. Cumulative error distribution
    sorted_errors = np.sort(analysis_df['percentage_error'])
    cumulative_pct = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    axes[1, 1].plot(sorted_errors, cumulative_pct, linewidth=2)
    axes[1, 1].axvline(x=18, color='red', linestyle='--', lw=2, label='18% Target')
    axes[1, 1].set_xlabel('Percentage Error (%)')
    axes[1, 1].set_ylabel('Cumulative Percentage of Predictions')
    axes[1, 1].set_title('Cumulative Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Top errors analysis
    top_errors = analysis_df.nlargest(20, 'percentage_error')
    axes[1, 2].scatter(top_errors['predicted_income'], top_errors['percentage_error'], 
                      s=50, color='red', alpha=0.7)
    axes[1, 2].set_xlabel('Predicted Income (₹)')
    axes[1, 2].set_ylabel('Percentage Error (%)')
    axes[1, 2].set_title('Top 20 Prediction Errors')
    axes[1, 2].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()

def save_results(analysis_df, method_name, output_dir='./'):
    """Save analysis results to CSV files"""
    
    # Ground truth CSV
    ground_truth_file = f"{output_dir}estimated_ground_truth_{method_name.lower()}.csv"
    ground_truth_df = analysis_df[['FarmerID', 'actual_income']].copy()
    ground_truth_df.to_csv(ground_truth_file, index=False)
    
    # Full analysis CSV
    full_analysis_file = f"{output_dir}full_analysis_{method_name.lower()}.csv"
    analysis_df.to_csv(full_analysis_file, index=False)
    
    # Error summary CSV
    error_summary_file = f"{output_dir}error_summary_{method_name.lower()}.csv"
    
    # Calculate summary statistics
    income_ranges = [(0, 2e5, '0-2L'), (2e5, 5e5, '2-5L'), (5e5, 10e5, '5-10L'), (10e5, float('inf'), '10L+')]
    summary_data = []
    
    for min_val, max_val, label in income_ranges:
        mask = (analysis_df['actual_income'] >= min_val) & (analysis_df['actual_income'] < max_val)
        if mask.sum() > 0:
            subset = analysis_df[mask]
            summary_data.append({
                'income_range': label,
                'count': len(subset),
                'mean_actual': subset['actual_income'].mean(),
                'mean_predicted': subset['predicted_income'].mean(),
                'mean_absolute_error': subset['absolute_error'].mean(),
                'mean_percentage_error': subset['percentage_error'].mean(),
                'median_percentage_error': subset['percentage_error'].median()
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(error_summary_file, index=False)
    
    print(f"\nFiles saved:")
    print(f"  Ground truth: {ground_truth_file}")
    print(f"  Full analysis: {full_analysis_file}")
    print(f"  Error summary: {error_summary_file}")

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main_analysis(submission_file_path, target_mape=19.18):
    """
    Main function to run complete analysis
    
    Parameters:
    submission_file_path: Path to your submission CSV file
    target_mape: Target MAPE to achieve (default: 19.18)
    """
    
    print("="*80)
    print("FARMER INCOME PREDICTION - TEST DATA ANALYSIS")
    print("="*80)
    
    # Load submission data
    submission_df = load_submission_data(submission_file_path)
    farmer_ids = submission_df['FarmerID'].values
    predicted_values = submission_df['predicted_income'].values
    
    print(f"\nLoaded {len(predicted_values)} predictions")
    print(f"Predicted income range: ₹{predicted_values.min():,.0f} - ₹{predicted_values.max():,.0f}")
    print(f"Target MAPE to reverse-engineer: {target_mape}%")
    
    # Method 1: Normal error distribution
    actual_values_1, mape_1 = estimate_ground_truth_method1(predicted_values, target_mape)
    analysis_df_1, metrics_1 = analyze_predictions(farmer_ids, actual_values_1, predicted_values)
    create_visualizations(analysis_df_1)
    save_results(analysis_df_1, "method1_normal")
    
    # Method 2: Income range-based errors
    actual_values_2, mape_2 = estimate_ground_truth_method2(predicted_values, target_mape)
    analysis_df_2, metrics_2 = analyze_predictions(farmer_ids, actual_values_2, predicted_values)
    create_visualizations(analysis_df_2)
    save_results(analysis_df_2, "method2_income_based")
    
    # Method 3: Iterative matching
    actual_values_3, mape_3 = estimate_ground_truth_method3(predicted_values, target_mape)
    analysis_df_3, metrics_3 = analyze_predictions(farmer_ids, actual_values_3, predicted_values)
    create_visualizations(analysis_df_3)
    save_results(analysis_df_3, "method3_iterative")
    
    # Summary comparison
    print("\n" + "="*80)
    print("METHOD COMPARISON SUMMARY")
    print("="*80)
    
    methods = [
        ("Method 1 (Normal Distribution)", mape_1, metrics_1),
        ("Method 2 (Income-Based)", mape_2, metrics_2),
        ("Method 3 (Iterative)", mape_3, metrics_3)
    ]
    
    for method_name, achieved_mape, metrics in methods:
        print(f"\n{method_name}:")
        print(f"  Achieved MAPE: {achieved_mape:.3f}%")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  Mean Abs Error: ₹{metrics['mean_abs_error']:,.0f}")
        print(f"  Median Abs Error: ₹{metrics['median_abs_error']:,.0f}")
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nNote: These are estimated ground truth values based on your known MAPE.")
    print("The actual test set ground truth may differ significantly.")
    print("Use these estimates for analysis purposes only.")
    
    return analysis_df_1, analysis_df_2, analysis_df_3

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

"""
USAGE INSTRUCTIONS:

1. Save your submission file as 'submission.csv' in the same directory
2. Run the analysis:

# Basic usage
analysis_results = main_analysis('submission.csv', target_mape=19.18)

# If your submission file has different column names, modify accordingly
# The function expects columns: 'FarmerID' and 'predicted_income'

3. The function will:
   - Generate 3 different estimated ground truth datasets
   - Create comprehensive visualizations
   - Save analysis results to CSV files
   - Provide detailed performance metrics

4. Output files will be created:
   - estimated_ground_truth_method1_normal.csv
   - estimated_ground_truth_method2_income_based.csv  
   - estimated_ground_truth_method3_iterative.csv
   - full_analysis_[method].csv (includes all metrics)
   - error_summary_[method].csv (performance by income range)

IMPORTANT LIMITATIONS:
- These are ESTIMATES based on statistical assumptions
- Real ground truth may be significantly different
- Multiple ground truth datasets could produce the same MAPE
- Use for analysis and learning purposes only
"""

# Uncomment to run analysis:
# analysis_results = main_analysis('submission.csv', target_mape=19.18)