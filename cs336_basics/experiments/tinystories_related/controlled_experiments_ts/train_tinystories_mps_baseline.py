from cs336_basics.training.training import train


def main():
    train(
        # data path
        train_path="data/tinystories_train.npy",
        val_path="data/tinystories_valid.npy",

        # Transformer parameters
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,

        # Apple Silicon (low resources)
        batch_size=32,
        max_iters=5000,
        device="mps",

        # AdamW
        learning_rate=3e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),

        # Learning Rate Scheduler
        min_lr=3e-5,
        warmup_iters=100,
        cosine_iters=5000,

        # validation and Checkpoint
        eval_interval=100,
        checkpoint_interval=1000,
        checkpoint_path="data/tinystories_mps_checkpoint.pt",

        # other hyperparameters
        rope_theta=10000.0,
        eps=1e-8,
        max_l2_norm=1.0,
        log_interval=10,

        # W&B
        run_name="tinystories-mps-baseline",
        project="cs336-assignment1",
    )


if __name__ == "__main__":
    main()
