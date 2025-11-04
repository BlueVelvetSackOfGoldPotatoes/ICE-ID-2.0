from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os

from .external_models import (
    ExternalTrainConfig,
    prepare_training_data,
    train_hf_classifier,
    predict_edges_with_hf,
    train_tfidf_classifier,
    predict_edges_with_tfidf,
    predict_edges_with_sbert,
    train_mlp_classifier,
    predict_edges_with_mlp,
    train_splink_model,
    predict_edges_with_splink,
    train_xgboost_model,
    predict_edges_with_xgboost,
    train_lightgbm_model,
    predict_edges_with_lightgbm,
    train_cross_encoder,
    predict_edges_with_crossencoder,
    train_dedupe_model,
    predict_edges_with_dedupe,
    train_recordlinkage_model,
    predict_edges_with_recordlinkage,
    train_random_forest_model,
    predict_edges_with_random_forest,
    train_gradient_boosting_model,
    predict_edges_with_gradient_boosting,
)


@dataclass(frozen=True)
class TrainingContext:
    """Context parameters for running one or more external model trainings.

    Args:
        people_csv (str): Path to people.csv.
        labels_csv (str): Path to labels.csv.
        run_shard_dir (str): Target output directory where model outputs should be written.
        sample_frac (float): Fraction of people rows to use for pair data preparation.
        epochs (int): Number of epochs for PLM fine-tuning models.

    Returns:
        TrainingContext: Immutable context object.
    """
    people_csv: str
    labels_csv: str
    run_shard_dir: str
    sample_frac: float = 1.0
    epochs: int = 1


def run_external_models(model_names: List[str], ctx: TrainingContext) -> Dict[str, Dict[str, str]]:
    """Runs selected external models and writes standardized outputs to the output directory.

    Args:
        model_names (List[str]): Model identifiers to run.
        ctx (TrainingContext): Training context with paths and options.

    Returns:
        Dict[str, Dict[str, str]]: Mapping from model short-name to output paths.
    """
    if not model_names:
        return {}
    
    outputs: Dict[str, Dict[str, str]] = {}
    os.makedirs(ctx.run_shard_dir, exist_ok=True)
    data_root = os.path.join(ctx.run_shard_dir, "_external_data")
    ds = prepare_training_data(ExternalTrainConfig(
        people_csv=ctx.people_csv,
        labels_csv=ctx.labels_csv,
        out_root=data_root,
        sample_frac=ctx.sample_frac,
    ))

    if "Ditto (HF)" in model_names or "Ditto" in model_names or "ditto_hf" in model_names:
        out_dir = os.path.join(ctx.run_shard_dir, "ditto_hf")
        os.makedirs(out_dir, exist_ok=True)
        epochs = getattr(ctx, "ditto_epochs", 3)
        learning_rate = getattr(ctx, "ditto_learning_rate", 2e-5)
        batch_size = getattr(ctx, "ditto_batch_size", 16)
        max_length = getattr(ctx, "ditto_max_length", 512)
        model_dir = train_hf_classifier(ds, out_dir, epochs=max(1, int(epochs)))
        e, c = predict_edges_with_hf(ds, ctx.people_csv, out_dir, threshold=0.5)
        outputs["ditto_hf"] = {"model": model_dir, "edges": e, "clusters": c}

    if "ZeroER (SBERT)" in model_names or "zeroer_sbert" in model_names or "sbert" in model_names:
        out_dir = os.path.join(ctx.run_shard_dir, "zeroer_sbert")
        os.makedirs(out_dir, exist_ok=True)
        e, c = predict_edges_with_sbert(ds, ctx.people_csv, out_dir, threshold=0.7)
        outputs["zeroer_sbert"] = {"edges": e, "clusters": c}

    if "TF-IDF" in model_names or "tfidf" in model_names:
        out_dir = os.path.join(ctx.run_shard_dir, "tfidf")
        os.makedirs(out_dir, exist_ok=True)
        model_path = train_tfidf_classifier(ds, out_dir)
        e, c = predict_edges_with_tfidf(ds, ctx.people_csv, out_dir, threshold=0.5)
        outputs["tfidf"] = {"model": model_path, "edges": e, "clusters": c}

    if "MLP" in model_names or "mlp" in model_names:
        out_dir = os.path.join(ctx.run_shard_dir, "mlp")
        os.makedirs(out_dir, exist_ok=True)
        hidden_dim = getattr(ctx, "mlp_hidden_dim", 128)
        num_layers = getattr(ctx, "mlp_num_layers", 2)
        dropout = getattr(ctx, "mlp_dropout", 0.1)
        learning_rate = getattr(ctx, "mlp_learning_rate", 1e-3)
        batch_size = getattr(ctx, "mlp_batch_size", 32)
        epochs = getattr(ctx, "mlp_epochs", 10)
        model_dir = train_mlp_classifier(
            ds, out_dir,
            epochs=max(1, int(epochs)),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
            learning_rate=float(learning_rate),
            batch_size=int(batch_size),
        )
        e, c = predict_edges_with_mlp(ds, ctx.people_csv, out_dir, threshold=0.5)
        outputs["mlp"] = {"model": model_dir, "edges": e, "clusters": c}

    if "Splink" in model_names or "splink" in model_names:
        out_dir = os.path.join(ctx.run_shard_dir, "splink")
        os.makedirs(out_dir, exist_ok=True)
        threshold = getattr(ctx, "splink_threshold", 0.5)
        model_path = train_splink_model(ds, out_dir, threshold=float(threshold))
        e, c = predict_edges_with_splink(ds, ctx.people_csv, out_dir, threshold=float(threshold))
        outputs["splink"] = {"model": model_path, "edges": e, "clusters": c}

    if "XGBoost" in model_names or "xgboost" in model_names or "XGB" in model_names:
        out_dir = os.path.join(ctx.run_shard_dir, "xgboost")
        os.makedirs(out_dir, exist_ok=True)
        n_estimators = getattr(ctx, "xgb_n_estimators", 100)
        max_depth = getattr(ctx, "xgb_max_depth", 5)
        learning_rate = getattr(ctx, "xgb_learning_rate", 0.1)
        min_samples_split = getattr(ctx, "xgb_min_samples_split", 2)
        subsample = getattr(ctx, "xgb_subsample", 1.0)
        model_path = train_xgboost_model(
            ds, out_dir,
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            min_samples_split=int(min_samples_split),
            subsample=float(subsample)
        )
        e, c = predict_edges_with_xgboost(ds, ctx.people_csv, out_dir, threshold=0.5)
        outputs["xgboost"] = {"model": model_path, "edges": e, "clusters": c}

    if "LightGBM" in model_names or "lightgbm" in model_names or "LGBM" in model_names:
        out_dir = os.path.join(ctx.run_shard_dir, "lightgbm")
        os.makedirs(out_dir, exist_ok=True)
        n_estimators = getattr(ctx, "lgbm_n_estimators", 100)
        max_depth = getattr(ctx, "lgbm_max_depth", 5)
        learning_rate = getattr(ctx, "lgbm_learning_rate", 0.1)
        min_samples_split = getattr(ctx, "lgbm_min_samples_split", 2)
        subsample = getattr(ctx, "lgbm_subsample", 1.0)
        model_path = train_lightgbm_model(
            ds, out_dir,
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            min_samples_split=int(min_samples_split),
            subsample=float(subsample)
        )
        e, c = predict_edges_with_lightgbm(ds, ctx.people_csv, out_dir, threshold=0.5)
        outputs["lightgbm"] = {"model": model_path, "edges": e, "clusters": c}

    if "Cross-Encoder" in model_names or "CrossEncoder" in model_names or "crossencoder" in model_names:
        out_dir = os.path.join(ctx.run_shard_dir, "crossencoder")
        os.makedirs(out_dir, exist_ok=True)
        epochs = getattr(ctx, "crossencoder_epochs", 3)
        learning_rate = getattr(ctx, "crossencoder_learning_rate", 2e-5)
        batch_size = getattr(ctx, "crossencoder_batch_size", 16)
        max_length = getattr(ctx, "crossencoder_max_length", 512)
        model_dir = train_cross_encoder(ds, out_dir, epochs=max(1, int(epochs)))
        e, c = predict_edges_with_crossencoder(ds, ctx.people_csv, out_dir, threshold=0.5)
        outputs["crossencoder"] = {"model": model_dir, "edges": e, "clusters": c}

    if "Dedupe" in model_names or "dedupe" in model_names:
        out_dir = os.path.join(ctx.run_shard_dir, "dedupe")
        os.makedirs(out_dir, exist_ok=True)
        sample_size = getattr(ctx, "dedupe_sample_size", 15000)
        threshold = getattr(ctx, "dedupe_threshold", 0.5)
        model_path = train_dedupe_model(ds, out_dir, sample_size=int(sample_size))
        e, c = predict_edges_with_dedupe(ds, ctx.people_csv, out_dir, threshold=float(threshold))
        outputs["dedupe"] = {"model": model_path, "edges": e, "clusters": c}

    if "RecordLinkage" in model_names or "recordlinkage" in model_names:
        out_dir = os.path.join(ctx.run_shard_dir, "recordlinkage")
        os.makedirs(out_dir, exist_ok=True)
        model_path = train_recordlinkage_model(ds, out_dir)
        threshold = getattr(ctx, "rl_threshold", 0.5)
        e, c = predict_edges_with_recordlinkage(ds, ctx.people_csv, out_dir, threshold=float(threshold))
        outputs["recordlinkage"] = {"model": model_path, "edges": e, "clusters": c}

    if "TF-IDF" in model_names or "tfidf" in model_names:
        out_dir = os.path.join(ctx.run_shard_dir, "tfidf")
        os.makedirs(out_dir, exist_ok=True)
        max_features = getattr(ctx, "tfidf_max_features", 20000)
        lr_C = getattr(ctx, "lr_C", 1.0)
        model_path = train_tfidf_classifier(ds, out_dir, max_features=int(max_features), lr_C=float(lr_C))
        e, c = predict_edges_with_tfidf(ds, ctx.people_csv, out_dir, threshold=0.5)
        outputs["tfidf"] = {"model": model_path, "edges": e, "clusters": c}

    if "Random Forest" in model_names or "random_forest" in model_names or "rf" in [m.lower() for m in model_names]:
        out_dir = os.path.join(ctx.run_shard_dir, "random_forest")
        os.makedirs(out_dir, exist_ok=True)
        n_estimators = getattr(ctx, "rf_n_estimators", 200)
        max_depth = getattr(ctx, "rf_max_depth", 10)
        min_samples_split = getattr(ctx, "rf_min_samples_split", 5)
        min_samples_leaf = getattr(ctx, "rf_min_samples_leaf", 2)
        max_features = getattr(ctx, "rf_max_features", "sqrt")
        model_path = train_random_forest_model(
            ds, out_dir,
            n_estimators=int(n_estimators),
            max_depth=int(max_depth) if max_depth else None,
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            max_features=max_features
        )
        e, c = predict_edges_with_random_forest(ds, ctx.people_csv, out_dir, threshold=0.5)
        outputs["random_forest"] = {"model": model_path, "edges": e, "clusters": c}

    if "Gradient Boosting" in model_names or "gradient_boosting" in model_names or "gb" in [m.lower() for m in model_names]:
        out_dir = os.path.join(ctx.run_shard_dir, "gradient_boosting")
        os.makedirs(out_dir, exist_ok=True)
        n_estimators = getattr(ctx, "gb_n_estimators", 200)
        learning_rate = getattr(ctx, "gb_learning_rate", 0.05)
        max_depth = getattr(ctx, "gb_max_depth", 6)
        min_samples_split = getattr(ctx, "gb_min_samples_split", 10)
        min_samples_leaf = getattr(ctx, "gb_min_samples_leaf", 4)
        subsample = getattr(ctx, "gb_subsample", 0.8)
        model_path = train_gradient_boosting_model(
            ds, out_dir,
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            subsample=float(subsample)
        )
        e, c = predict_edges_with_gradient_boosting(ds, ctx.people_csv, out_dir, threshold=0.5)
        outputs["gradient_boosting"] = {"model": model_path, "edges": e, "clusters": c}

    return outputs


