"""Train ComplEx or relation-aware GNN link predictors on NeuroOracle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from models.common.artifacts import RunArtifacts
from models.kg_link_prediction.gnn import (
    GNNLinkPredictor,
    TripleIndex,
    message_graph,
    sample_negatives,
)
from neurooracle.src.kge.complex_scorer import ComplExScorer, TrainConfig
from neurooracle.src.kge.triple_loader import load_triples_from_kg, split_triples


def _filtered_ranking_metrics(model, embeddings, triples, known, limit=200):
    ranks = []
    for source, relation, target in triples[:limit].tolist():
        relation_vector = model.relation_embedding.weight[relation]
        scores = (
            embeddings[source][None, :] * relation_vector[None, :] * embeddings
        ).sum(dim=1)
        for known_source, known_relation, known_target in known:
            if (
                known_source == source
                and known_relation == relation
                and known_target != target
            ):
                scores[known_target] = -torch.inf
        target_score = scores[target]
        ranks.append(1 + int((scores > target_score).sum().item()))
    if not ranks:
        return {"mrr": 0.0, "hits_at_1": 0.0, "hits_at_3": 0.0, "hits_at_10": 0.0}
    values = np.asarray(ranks)
    return {
        "mrr": float(np.mean(1 / values)),
        "hits_at_1": float(np.mean(values <= 1)),
        "hits_at_3": float(np.mean(values <= 3)),
        "hits_at_10": float(np.mean(values <= 10)),
    }


def _train_gnn(args, train, validation, test):
    device = torch.device(args.device)
    all_triples = [*train, *validation, *test]
    index = TripleIndex.from_triples(all_triples)
    train_ids = index.encode(train)
    validation_ids = index.encode(validation)
    test_ids = index.encode(test)
    edge_index, edge_type = message_graph(train_ids)
    model = GNNLinkPredictor(
        len(index.entity_to_id),
        len(index.relation_to_id),
        args.model,
        args.embedding_dim,
        args.layers,
        args.dropout,
    ).to(device)
    train_ids_device = train_ids.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    known = {tuple(row) for row in index.encode(all_triples).tolist()}
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(args.epochs):
        model.train()
        negative = sample_negatives(
            train_ids,
            len(index.entity_to_id),
            known,
            args.negatives,
            rng,
        )
        embeddings = model.encode(edge_index, edge_type)
        positive_score = model.score(embeddings, train_ids_device)
        negative_score = model.score(embeddings, negative.to(device))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            positive_score, torch.ones_like(positive_score)
        ) + torch.nn.functional.binary_cross_entropy_with_logits(
            negative_score, torch.zeros_like(negative_score)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    model.eval()
    evaluation = (
        test_ids
        if len(test_ids)
        else validation_ids
        if len(validation_ids)
        else train_ids[: min(100, len(train_ids))]
    )
    negative = sample_negatives(
        evaluation,
        len(index.entity_to_id),
        known,
        args.negatives,
        rng,
    )
    with torch.no_grad():
        embeddings = model.encode(edge_index, edge_type)
        positive_score = model.score(embeddings, evaluation.to(device)).cpu().numpy()
        negative_score = model.score(embeddings, negative.to(device)).cpu().numpy()
    labels = np.concatenate([np.ones(len(positive_score)), np.zeros(len(negative_score))])
    scores = np.concatenate([positive_score, negative_score])
    metrics = {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "final_loss": history[-1],
        "entities": len(index.entity_to_id),
        "relations": len(index.relation_to_id),
    }
    metrics.update(
        _filtered_ranking_metrics(
            model, embeddings, evaluation.to(device), known
        )
    )
    return model, index, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg", required=True)
    parser.add_argument("--model", choices=["complex", "rgcn", "graphsage", "gat"], required=True)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--negatives", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--min-confidence", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    triples, domains = load_triples_from_kg(args.kg, args.min_confidence)
    train, validation, test = split_triples(triples, domains, seed=args.seed)
    output = Path(args.output_dir)
    artifacts = RunArtifacts(output, vars(args))
    artifacts.write_config()
    if args.model == "complex":
        scorer = ComplExScorer(dim=args.embedding_dim)
        history = scorer.fit(
            train,
            validation,
            TrainConfig(
                embedding_dim=args.embedding_dim,
                epochs=args.epochs,
                learning_rate=args.lr,
                negatives_per_pos=args.negatives,
                weight_decay=args.weight_decay,
                device=args.device,
            ),
        )
        metrics = {
            "test_auroc": scorer.auroc(test),
            "final_loss": float(history["loss"][-1]),
            "entities": len(scorer.ent2idx),
            "relations": len(scorer.rel2idx),
        }
        scorer.save(output / "checkpoint.pt")
    else:
        model, index, metrics = _train_gnn(args, train, validation, test)
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "entity_to_id": index.entity_to_id,
                "relation_to_id": index.relation_to_id,
                "config": vars(args),
            },
            output / "checkpoint.pt",
        )
    artifacts.write_metrics(metrics)
    artifacts.write_manifest(
        {"train_triples": len(train), "validation_triples": len(validation), "test_triples": len(test)}
    )
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
