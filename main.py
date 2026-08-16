def main() -> None:
    import argparse
    import torch
    from model import MatryoshkaEncoder
    from preprocess import CharacterTokenizer
    from train import Trainer
    from utils import DEFAULT_PAIRS, PairDataset, create_dataloader, evaluate_recall_at_one, load_pairs, set_seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--dimensions", default="8,16,32,64")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    args = parser.parse_args()
    dimensions = [int(value) for value in args.dimensions.split(",") if value]
    if not dimensions or max(dimensions) > args.embedding_dim or min(dimensions) <= 0:
        parser.error("dimensions must be positive and no greater than embedding-dim")
    set_seed(args.seed)
    pairs = load_pairs(__import__("pathlib").Path(args.data_path)) if args.data_path else DEFAULT_PAIRS
    tokenizer = CharacterTokenizer().fit([text for pair in pairs for text in pair])
    dataset = PairDataset(pairs, tokenizer, args.max_length)
    device = torch.device(args.device)
    model = MatryoshkaEncoder(tokenizer.vocabulary_size, args.embedding_dim, args.num_heads, args.num_layers, args.max_length)
    trainer = Trainer(model, dimensions, 0.1, 1e-3, device)
    loader = create_dataloader(dataset, args.batch_size, True)
    for epoch in range(1, args.epochs + 1):
        print(f"epoch={epoch} loss={trainer.train_epoch(loader):.4f}")
    evaluation = create_dataloader(dataset, args.batch_size, False)
    model.eval()
    with torch.no_grad():
        query_embeddings, positive_embeddings = [], []
        for batch in evaluation:
            query_embeddings.append(model(batch["query_ids"].to(device), batch["query_mask"].to(device)).cpu())
            positive_embeddings.append(model(batch["positive_ids"].to(device), batch["positive_mask"].to(device)).cpu())
    queries, positives = torch.cat(query_embeddings), torch.cat(positive_embeddings)
    for dimension in dimensions:
        print(f"dimension={dimension} Recall@1={evaluate_recall_at_one(queries[:, :dimension], positives[:, :dimension]):.4f}")


if __name__ == "__main__":
    main()