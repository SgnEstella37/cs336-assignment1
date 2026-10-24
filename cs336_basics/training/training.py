# fulfill the whole training loop
import numpy as np
import torch
from cs336_basics.model.transformer import TransformerLM
from cs336_basics.training.AdamW_optimizer import AdamW
from cs336_basics.training.lr_scheduler import learning_rate_scheduling
from cs336_basics.training.data_loader import get_batch
from cs336_basics.training.loss import cross_entropy
from cs336_basics.training.gradient_clipping import gradient_clipping
from cs336_basics.training.check_point import save_checkpoint


def train(
    train_path,
    val_path,
    vocab_size,
    context_length,
    d_model,
    num_layers,
    num_heads,
    d_ff,
    batch_size,
    max_iters,
    device,
    learning_rate,
    weight_decay,
    betas: tuple[float, float],
    min_lr,
    warmup_iters,
    cosine_iters,
    eval_interval,
    checkpoint_interval,
    checkpoint_path,
    rope_theta=10000.0,
    eps=1e-8,
    max_l2_norm=1.0,
):
    # initialize model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    ).to(device)

    # initialize optimizer
    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
    )

    # load the train and validation data
    train_data = np.load(train_path, mmap_mode="r")
    val_data = np.load(val_path, mmap_mode="r")

    for t in range(max_iters):

        # apply learning rate scheduling
        current_lr = learning_rate_scheduling(
            t=t,
            alpha_max=learning_rate,
            alpha_min=min_lr,
            T_w=warmup_iters,
            T_c=cosine_iters,
        )

        for group in optimizer.param_groups:
            group["lr"] = current_lr

        inputs, targets = get_batch(
            x=train_data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        # train
        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy(
            logits=logits.reshape(-1, vocab_size), target=targets.reshape(-1)
        )
        loss.backward()
        gradient_clipping(parameters=list(model.parameters()), max_l2_norm=max_l2_norm)
        optimizer.step()

        # validation
        if (t + 1) % eval_interval == 0:
            model.eval()

            with torch.no_grad():
                val_inputs, val_targets = get_batch(
                    x=val_data,
                    batch_size=batch_size,
                    context_length=context_length,
                    device=device,
                )

                val_logits = model(val_inputs)
                val_loss = cross_entropy(
                    logits=val_logits.reshape(-1, vocab_size),
                    target=val_targets.reshape(-1),
                )

                print(
                    f"step={t+1}, "
                    f"train_loss={loss.item():.4f}, "
                    f"val_loss={val_loss.item():.4f}"
                )

            model.train()

        # save checkpoint
        if (t + 1) % checkpoint_interval == 0:
            save_checkpoint(
                model=model, optimizer=optimizer, iteration=t + 1, out=checkpoint_path
            )

#launch!
if __name__=="__main__":
    pass
    