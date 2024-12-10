from datetime import datetime
from functools import partial
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from piano_metrics.piano_metric import (
    F1Metric,
    MetricsRunner,
    KeyCorrelationMetric,
    PitchCorrelationMetric,
    DstartCorrelationMetric,
    DurationCorrelationMetric,
    VelocityCorrelationMetric,
)

import piano_generation.database.database_manager as database_manager


def create_metrics_runner() -> MetricsRunner:
    """Create metrics runner with all metrics"""
    metrics = [
        F1Metric(use_pitch_class=True, velocity_threshold=30, min_time_unit=0.01),
        KeyCorrelationMetric(segment_duration=0.125, use_weighted=True),
        PitchCorrelationMetric(use_weighted=True),
        VelocityCorrelationMetric(use_weighted=True),
        DstartCorrelationMetric(n_bins=50),
        DurationCorrelationMetric(n_bins=50),
    ]
    return MetricsRunner(metrics)


def validate_dataframes_and_results(
    source_df: pd.DataFrame,
    prompt_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    results: Dict,
) -> Tuple[bool, str]:
    """Validate dataframes and metric results"""
    # Check for empty dataframes
    if source_df.empty:
        return False, "Source dataframe is empty"
    if prompt_df.empty:
        return False, "Prompt dataframe is empty"
    if generated_df.empty:
        return False, "Generated dataframe is empty"

    # Check for required columns
    required_columns = ["pitch", "start", "end"]
    for df, name in [(source_df, "source"), (prompt_df, "prompt"), (generated_df, "generated")]:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing columns {missing_cols} in {name} dataframe"

    # Check for NaN values in results
    if results is None:
        return False, "Metrics calculation failed"

    for metric_name, result in results.items():
        if pd.isna(result.value):
            return False, f"NaN value in {metric_name}"

    return True, ""


def process_generation(generation_data: tuple, metrics_runner: MetricsRunner) -> Dict:
    """Process a single generation"""
    try:
        generation, model_info = generation_data

        # Convert JSON notes to DataFrames
        prompt_df = pd.DataFrame(generation["prompt_notes"])
        generated_df = pd.DataFrame(generation["generated_notes"])

        # Get source notes
        source = database_manager.get_source(generation["source_id"])
        source_df = pd.DataFrame(source.iloc[0]["notes"])

        # Calculate metrics
        results = metrics_runner.calculate_all(prompt_df, generated_df)

        # Validate dataframes and results
        is_valid, error_message = validate_dataframes_and_results(source_df, prompt_df, generated_df, results)

        if not is_valid:
            return {
                "error": True,
                "generation_id": generation["generation_id"],
                "error_message": error_message,
                "model_name": model_info["name"],
                "model_size_M": model_info["milion_parameters"],
                "total_tokens": model_info["total_tokens"],
            }

        # Extract composer and title
        source_info = generation["source"]
        composer = source_info.get("composer", "Unknown")
        title = source_info.get("title", "Unknown")

        # Compile metrics
        metrics_dict = {metric_name: result.value for metric_name, result in results.items()}

        # Extract additional metrics from metadata
        detailed_metrics = {}
        for metric_name, result in results.items():
            if result.metadata and "detailed_metrics" in result.metadata:
                for key, value in result.metadata["detailed_metrics"].items():
                    detailed_metrics[f"{metric_name}_{key}"] = value

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
            **metrics_dict,
            **detailed_metrics,
            "generator_parameters": str(generation["generator_parameters"]),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_notes_count": len(source_df),
            "generated_notes_count": len(generated_df),
        }

        return result

    except Exception as e:
        return {
            "error": True,
            "generation_id": generation.get("generation_id", "unknown"),
            "error_message": str(e),
            "model_name": model_info["name"],
            "model_size_M": model_info["milion_parameters"],
            "total_tokens": model_info["total_tokens"],
        }


def analyze_model_generations(metrics_runner: MetricsRunner) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze all generations across all models"""
    print("\n1. Fetching models from database...")
    models_df = database_manager.select_models_with_generations()
    print(f"Found {len(models_df)} models with generations")

    # Initialize metrics names from runner
    metric_names = metrics_runner.list_metrics()
    print(f"Using metrics: {metric_names}")

    results = []
    errors = []

    # Count total generations
    print("\n2. Counting total generations...")
    total_generations = sum(
        len(database_manager.get_model_predictions({"model_id": model["model_id"]})) for _, model in models_df.iterrows()
    )

    print(f"Total generations to process: {total_generations}")
    print(f"Using {cpu_count()} CPU cores")

    # Create progress bars
    model_pbar = tqdm(total=len(models_df), desc="Models", position=0)
    generation_pbar = tqdm(total=total_generations, desc="Generations", position=1)

    # Process generations
    with Pool(processes=cpu_count(), initializer=_init_metrics_runner, initargs=(metrics_runner,)) as pool:
        for _, model in models_df.iterrows():
            predictions_df = database_manager.get_model_predictions(model_filters={"model_id": model["model_id"]})

            generation_data = [(row, model) for _, row in predictions_df.iterrows()]
            batch_results = pool.imap(process_generation_wrapper, generation_data)

            for result in batch_results:
                if result.get("error", False):
                    errors.append(result)
                else:
                    results.append(result)
                generation_pbar.update(1)

            model_pbar.update(1)

    model_pbar.close()
    generation_pbar.close()

    # Handle results and errors
    _handle_processing_results(results, errors)

    # Create DataFrame
    results_df = pd.DataFrame(results)
    if len(results_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Add base model name
    results_df["base_model_name"] = results_df["model_name"].apply(_extract_base_model_name)

    print("Calculating aggregate metrics...")
    # Calculate aggregates using metric names from runner
    return results_df, _calculate_aggregations(results_df, metric_names)


def _init_metrics_runner(runner):
    """Initialize metrics runner for each process"""
    global _metrics_runner
    _metrics_runner = runner


def process_generation_wrapper(generation_data):
    """Wrapper to use global metrics runner"""
    global _metrics_runner
    return process_generation(generation_data, _metrics_runner)


def _extract_base_model_name(model_name: str) -> str:
    """Extract base model name removing timestamps and indicators"""
    return "-".join(part for part in model_name.split("-") if not any(c.isdigit() for c in part) and "last" not in part)


def _calculate_aggregations(results_df: pd.DataFrame, metric_names: List[str]) -> pd.DataFrame:
    """Calculate aggregate metrics"""
    # Group by relevant columns
    agg_metrics = (
        results_df.groupby(
            ["base_model_name", "model_name", "model_size_M", "total_tokens", "best_val_loss", "train_loss", "training_task"]
        )
        .agg(
            {
                **{name: ["mean", "std", "count", "sem"] for name in metric_names},
                "source_notes_count": ["mean", "std"],
                "generated_notes_count": ["mean", "std"],
                "composer": "nunique",
                "title": "nunique",
            }
        )
        .reset_index()
    )

    # Flatten column names once
    agg_metrics.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in agg_metrics.columns]

    # Add confidence intervals and improvements
    agg_metrics = _add_statistical_metrics(agg_metrics, metric_names)

    return agg_metrics


def _add_statistical_metrics(df: pd.DataFrame, metric_names: List[str]) -> pd.DataFrame:
    """Add confidence intervals and improvement metrics"""
    # Add confidence intervals (95%)
    for metric in metric_names:
        if f"{metric}_sem" in df.columns:
            df[f"{metric}_ci_lower"] = df[f"{metric}_mean"] - 1.96 * df[f"{metric}_sem"]
            df[f"{metric}_ci_upper"] = df[f"{metric}_mean"] + 1.96 * df[f"{metric}_sem"]

    # Add tokens in millions
    df["tokens_M"] = df["total_tokens"] / 1_000_000

    # Sort
    df = df.sort_values(["model_size_M", "tokens_M", "best_val_loss"])

    # Calculate improvements
    for metric in metric_names:
        df[f"{metric}_improvement"] = (
            df.groupby("base_model_name")
            .apply(lambda x: (x[f"{metric}_mean"] - x[f"{metric}_mean"].iloc[0]) / x[f"{metric}_mean"].iloc[0])
            .reset_index(level=0, drop=True)
        )

    return df


def _handle_processing_results(results: List[Dict], errors: List[Dict]):
    """Handle and report processing results"""
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {len(results)} generations")

    if errors:
        print("\nError summary:")
        error_counts = {}
        for error in errors:
            error_type = error["error_message"]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        for error_type, count in error_counts.items():
            print(f"- {error_type}: {count} occurrences")

        error_df = pd.DataFrame(errors)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_df.to_csv(f"processing_errors_{timestamp}.csv", index=False)
        print("\nDetailed error log saved to processing_errors_*.csv")


def main():
    """Main function to run the analysis and save results"""
    print("=" * 50)
    print("Starting MIDI Model Analysis")
    print("=" * 50)

    start_time = datetime.now()
    metrics_runner = create_metrics_runner()
    # Run analysis
    detailed_results, aggregate_results = analyze_model_generations(metrics_runner)

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
    metrics = metrics_runner.list_metrics()

    for base_model in sorted(aggregate_results["base_model_name"].unique()):
        model_data = aggregate_results[aggregate_results["base_model_name"] == base_model]
        print(f"\nModel: {base_model} ({model_data['model_size_M'].iloc[0]}M)")
        print(f"Training progress: {model_data['tokens_M'].min():.1f}M -> {model_data['tokens_M'].max():.1f}M tokens")

        for metric in metrics:
            initial = model_data[f"{metric}_mean"].iloc[0]
            final = model_data[f"{metric}_mean"].iloc[-1]
            improvement = (final - initial) / initial * 100
            print(f"- {metric}: {initial:.3f} -> {final:.3f} ({improvement:+.1f}%)")

    # Calculate correlations
    print("\nCorrelations between model parameters and metrics:")
    print("-" * 50)
    params = ["model_size_M", "tokens_M", "best_val_loss"]

    for param in params:
        print(f"\n{param}:")
        for metric in metrics:
            corr = aggregate_results[param].corr(aggregate_results[f"{metric}_mean"])
            print(f"  vs {metric}: {corr:.3f}")


if __name__ == "__main__":
    main()
