import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from piano_metrics import f1_piano, key_distribution, pitch_distribution
import piano_generation.database.database_manager as database_manager

def validate_dataframes_and_metrics(source_df: pd.DataFrame, 
                                  prompt_df: pd.DataFrame, 
                                  generated_df: pd.DataFrame,
                                  metrics: Dict) -> Tuple[bool, str]:
    """
    Validate dataframes and metrics for processing.
    
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    # Check for empty dataframes
    if source_df.empty:
        return False, "Source dataframe is empty"
    if prompt_df.empty:
        return False, "Prompt dataframe is empty"
    if generated_df.empty:
        return False, "Generated dataframe is empty"
    
    # Check for required columns
    required_columns = ['pitch', 'start', 'end']
    for df, name in [(source_df, 'source'), (prompt_df, 'prompt'), (generated_df, 'generated')]:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing columns {missing_cols} in {name} dataframe"
    
    # Check for NaN values in metrics
    if metrics is None:
        return False, "Metrics calculation failed"
    
    metric_values = [
        metrics['key_correlation'],
        metrics['pitch_correlation'],
        metrics['f1_score'],
        metrics['precision'],
        metrics['recall']
    ]
    
    if any(pd.isna(v) for v in metric_values):
        return False, "NaN values in metrics"
        
    return True, ""

def process_generation(generation_data: tuple) -> Dict:
    """Process a single generation (used for parallel processing)"""
    try:
        generation, model_info = generation_data
        
        # Convert JSON notes to DataFrames
        prompt_df = pd.DataFrame(generation['prompt_notes'])
        generated_df = pd.DataFrame(generation['generated_notes'])
        
        # Get source notes
        source = database_manager.get_source(generation['source_id'])
        source_df = pd.DataFrame(source.iloc[0]['notes'])
        
        # Calculate metrics
        metrics = calculate_sample_metrics(source_df, generated_df)
        
        # Validate dataframes and metrics
        is_valid, error_message = validate_dataframes_and_metrics(
            source_df, prompt_df, generated_df, metrics
        )
        
        if not is_valid:
            return {
                "error": True,
                "generation_id": generation['generation_id'],
                "error_message": error_message,
                "model_name": model_info["name"],
                "model_size_M": model_info["milion_parameters"],
                "total_tokens": model_info["total_tokens"]
            }
        
        # Extract composer and title from source if available
        source_info = generation['source']
        composer = source_info.get('composer', 'Unknown')
        title = source_info.get('title', 'Unknown')
        
        # Compile result row
        result = {
            "model_name": model_info["name"],
            "model_size_M": model_info["milion_parameters"],
            "total_tokens": model_info["total_tokens"],
            "best_val_loss": model_info["best_val_loss"],
            "train_loss": model_info["train_loss"],
            "training_task": model_info["training_task"],
            "generator_name": generation["generator_name"],
            "task": generation["task"],
            "composer": composer,
            "title": title,
            "generation_id": generation["generation_id"],
            **metrics,
            "generator_parameters": str(generation["generator_parameters"]),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_notes_count": len(source_df),
            "generated_notes_count": len(generated_df)
        }
        
        return result
    
    except Exception as e:
        return {
            "error": True,
            "generation_id": generation.get('generation_id', 'unknown'),
            "error_message": str(e),
            "model_name": model_info["name"],
            "model_size_M": model_info["milion_parameters"],
            "total_tokens": model_info["total_tokens"]
        }

def analyze_model_generations() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze all generations across all models and compile results"""
    
    print("\n1. Fetching models from database...")
    models_df = database_manager.select_models_with_generations()
    print(f"Found {len(models_df)} models with generations")
    
    results = []
    errors = []
    total_generations = 0
    skipped_generations = 0
    
    # First pass to count total generations
    print("\n2. Counting total generations...")
    for _, model in models_df.iterrows():
        predictions = database_manager.get_model_predictions(
            model_filters={"model_id": model['model_id']}
        )
        total_generations += len(predictions)
    
    print(f"Total generations to process: {total_generations}")
    print(f"Using {cpu_count()} CPU cores for parallel processing")
    
    # Create progress bars
    model_pbar = tqdm(total=len(models_df), desc="Models processed", position=0)
    generation_pbar = tqdm(total=total_generations, desc="Generations analyzed", position=1)
    
    print("\n3. Starting parallel analysis...")
    
    # Process each model's generations in parallel
    with Pool(processes=cpu_count()) as pool:
        for _, model in models_df.iterrows():
            model_name = model['name']
            model_tokens = model['total_tokens']
            
            tqdm.write(f"\nProcessing model: {model_name} ({model['milion_parameters']}M params, {model_tokens/1e6:.1f}M tokens)")
            
            # Get all generations for this model
            predictions_df = database_manager.get_model_predictions(
                model_filters={"model_id": model['model_id']}
            )
            
            tqdm.write(f"Found {len(predictions_df)} generations for this model")
            
            # Prepare data for parallel processing
            generation_data = [(row, model) for _, row in predictions_df.iterrows()]
            
            # Process generations in parallel
            batch_results = pool.imap(process_generation, generation_data)
            
            # Process results
            for result in batch_results:
                if result.get('error', False):
                    errors.append(result)
                    skipped_generations += 1
                else:
                    results.append(result)
                generation_pbar.update(1)
            
            model_pbar.update(1)
    
    model_pbar.close()
    generation_pbar.close()
    
    # Report processing statistics
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {len(results)} generations")
    print(f"Skipped/Error: {skipped_generations} generations")
    
    # Handle errors
    if errors:
        print("\nError summary:")
        error_counts = {}
        for error in errors:
            error_type = error['error_message']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        for error_type, count in error_counts.items():
            print(f"- {error_type}: {count} occurrences")
        
        error_df = pd.DataFrame(errors)
        error_df.to_csv(f"processing_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
        print("\nDetailed error log saved to processing_errors_*.csv")
    
    print("\n4. Creating summary dataframes...")
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("No valid results to analyze!")
        return pd.DataFrame(), pd.DataFrame()
    
    # Extract base model name (remove timestamp and training type indicators)
    results_df['base_model_name'] = results_df['model_name'].apply(
        lambda x: '-'.join([part for part in x.split('-') 
                          if not any(c.isdigit() for c in part) 
                          and 'last' not in part])
    )
    
    print("Calculating aggregate metrics...")
    # Calculate aggregate statistics per unique model configuration
    # Group by base model name, size, tokens, and loss values to treat each checkpoint as separate
    agg_metrics = results_df.groupby([
        "base_model_name",
        "model_name",
        "model_size_M",
        "total_tokens",
        "best_val_loss",
        "train_loss",
        "training_task"
    ]).agg({
        "key_correlation": ["mean", "std", "count", "sem"],  # sem for standard error of mean
        "pitch_correlation": ["mean", "std", "sem"],
        "f1_score": ["mean", "std", "sem"],
        "precision": ["mean", "std", "sem"],
        "recall": ["mean", "std", "sem"],
        "source_notes_count": ["mean", "std"],
        "generated_notes_count": ["mean", "std"],
        "composer": "nunique",  # count unique composers
        "title": "nunique"      # count unique pieces
    }).reset_index()
    
    # Flatten column names
    agg_metrics.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] 
        for col in agg_metrics.columns
    ]
    
    # Add confidence intervals (95%)
    for metric in ['key_correlation', 'pitch_correlation', 'f1_score', 'precision', 'recall']:
        agg_metrics[f'{metric}_ci_lower'] = (
            agg_metrics[f'{metric}_mean'] - 1.96 * agg_metrics[f'{metric}_sem']
        )
        agg_metrics[f'{metric}_ci_upper'] = (
            agg_metrics[f'{metric}_mean'] + 1.96 * agg_metrics[f'{metric}_sem']
        )
    
    # Add tokens in millions for better readability
    agg_metrics['tokens_M'] = agg_metrics['total_tokens'] / 1_000_000
    
    # Sort by model size, tokens, and loss
    agg_metrics = agg_metrics.sort_values([
        'model_size_M', 
        'tokens_M', 
        'best_val_loss'
    ])
    
    # Calculate relative performance improvement for each base model
    for metric in ['key_correlation', 'pitch_correlation', 'f1_score']:
        agg_metrics[f'{metric}_improvement'] = agg_metrics.groupby('base_model_name').apply(
            lambda x: (x[f'{metric}_mean'] - x[f'{metric}_mean'].iloc[0]) / x[f'{metric}_mean'].iloc[0]
        ).reset_index(level=0, drop=True)
    
    return results_df, agg_metrics


def calculate_sample_metrics(
    prompt_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    params: Dict = {
        "segment_duration": 0.125,
        "use_weighted_key": True,
        "use_weighted_pitch": True,
        "min_time_unit": 0.01,
        "velocity_threshold": 30,
        "use_pitch_class": True
    }
) -> Dict:
    """Calculate metrics for a single generation sample"""
    
    key_corr, key_metrics = key_distribution.calculate_key_correlation(
        target_df=prompt_df,
        generated_df=generated_df,
        segment_duration=params["segment_duration"],
        use_weighted=params["use_weighted_key"]
    )
    
    pitch_corr, pitch_metrics = pitch_distribution.calculate_pitch_correlation(
        target_df=prompt_df,
        generated_df=generated_df,
        use_weighted=params["use_weighted_pitch"]
    )
    
    f1_score, f1_metrics = f1_piano.calculate_f1(
        target_df=prompt_df,
        generated_df=generated_df,
        min_time_unit=params["min_time_unit"],
        velocity_threshold=params["velocity_threshold"],
        use_pitch_class=params["use_pitch_class"]
    )
    
    return {
        "key_correlation": key_corr,
        "pitch_correlation": pitch_corr,
        "f1_score": f1_score,
        "top_target_keys": key_metrics["target_top_keys"][:3],
        "top_generated_keys": key_metrics["generated_top_keys"][:3],
        "target_active_pitches": pitch_metrics["target_active_pitches"],
        "generated_active_pitches": pitch_metrics["generated_active_pitches"],
        "precision": np.mean(f1_metrics["precision"]),
        "recall": np.mean(f1_metrics["recall"])
    }

def main():
    """Main function to run the analysis and save results"""
    print("=" * 50)
    print("Starting MIDI Model Analysis")
    print("=" * 50)
    
    start_time = datetime.now()
    
    # Run analysis
    detailed_results, aggregate_results = analyze_model_generations()
    
    if len(detailed_results) == 0:
        print("No results to save. Exiting.")
        return
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n5. Saving results...")
    
    # Save detailed results
    detailed_path = f"model_metrics_detailed_{timestamp}.csv"
    detailed_results.to_csv(detailed_path, index=False)
    print(f"Detailed results saved to: {detailed_path}")
    
    # Save aggregate results
    aggregate_path = f"model_metrics_aggregate_{timestamp}.csv"
    aggregate_results.to_csv(aggregate_path, index=False)
    print(f"Aggregate results saved to: {aggregate_path}")
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("Analysis Summary:")
    print("=" * 50)
    print(f"Total base models analyzed: {len(detailed_results['base_model_name'].unique())}")
    print(f"Total model checkpoints analyzed: {len(aggregate_results)}")
    print(f"Total generations analyzed: {len(detailed_results)}")
    print(f"Time taken: {datetime.now() - start_time}")
    
    # Create performance progression summary
    print("\nPerformance Progression Summary:")
    print("-" * 50)
    metrics = ['key_correlation', 'pitch_correlation', 'f1_score']
    
    for base_model in sorted(aggregate_results['base_model_name'].unique()):
        model_data = aggregate_results[aggregate_results['base_model_name'] == base_model]
        print(f"\nModel: {base_model} ({model_data['model_size_M'].iloc[0]}M)")
        print(f"Training progress: {model_data['tokens_M'].min():.1f}M -> {model_data['tokens_M'].max():.1f}M tokens")
        
        for metric in metrics:
            initial = model_data[f'{metric}_mean'].iloc[0]
            final = model_data[f'{metric}_mean'].iloc[-1]
            improvement = (final - initial) / initial * 100
            print(f"- {metric}: {initial:.3f} -> {final:.3f} ({improvement:+.1f}%)")
    
    # Calculate correlations
    print("\nCorrelations between model parameters and metrics:")
    print("-" * 50)
    params = ['model_size_M', 'tokens_M', 'best_val_loss']
    metrics = ['key_correlation_mean', 'pitch_correlation_mean', 'f1_score_mean']
    
    for param in params:
        print(f"\n{param}:")
        for metric in metrics:
            corr = aggregate_results[param].corr(aggregate_results[metric])
            print(f"  vs {metric}: {corr:.3f}")

if __name__ == "__main__":
    main()