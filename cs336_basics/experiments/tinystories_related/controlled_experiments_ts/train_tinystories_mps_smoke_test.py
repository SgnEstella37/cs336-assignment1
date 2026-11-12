
from cs336_basics.training.training import train


def main():
    train(
        # Data
        train_path="data/tinystories_train.npy",
        val_path="data/tinystories_valid.npy",

        # Transformer architecture (same as baseline)
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,

        # Short MPS smoke test
        batch_size=4,
        max_iters=10,
        device="mps",

        # AdamW
        learning_rate=3e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),

        # Learning rate scheduler
        min_lr=3e-5,
        warmup_iters=2,
        cosine_iters=10,

        # Validation and checkpoint
        eval_interval=5,
        checkpoint_interval=10,
        checkpoint_path="data/tinystories_mps_smoke_test.pt",

        # Other parameters
        rope_theta=10000.0,
        eps=1e-8,
        max_l2_norm=1.0,
        log_interval=1,

        # W&B
        run_name="tinystories-mps-smoke-test",
        project="cs336-assignment1",
    )


if __name__ == "__main__":
    main()
