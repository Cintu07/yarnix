"""
Yarnix Language Model v7.0 — "Multi-Clock Phase"

Uses YarnixCellV4 with 4 timescale bands:
  - Ultra-fast neurons (persist=0.5) → character/word patterns
  - Fast neurons (persist=0.8) → phrase/sentence structure
  - Slow neurons (persist=0.95) → paragraph/topic tracking
  - Ultra-slow neurons (persist=0.999) → chapter-level memory

Like the brain using gamma, beta, alpha, and theta waves simultaneously.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from yarnix_cell import YarnixModelV4
from config import FULL_STATE


class YarnixLM(nn.Module):
    def __init__(self, vocab_size=128, embed_size=64, hidden_size=256):
        super(YarnixLM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.engine = YarnixModelV4(
            input_size=embed_size,
            hidden_size=hidden_size,
            output_size=vocab_size,
            num_layers=2,
            harmonics=[1, 2, 4, 8],
            quantization_strength=0.05,
            clock_speeds=(0.50, 0.80, 0.95, 0.999),
        )
        
    def forward(self, x):
        emb = self.embedding(x)
        logits = self.engine(emb, return_sequence=True)
        return logits

    @torch.no_grad()
    def generate(self, start_text, max_len=200, temperature=0.8, top_k=15):
        self.eval()
        device = next(self.parameters()).device
        chars = [min(ord(c), self.vocab_size - 1) for c in start_text]
        generated = list(start_text)
        
        h = [torch.zeros(1, self.hidden_size, device=device)
             for _ in range(self.engine.num_layers)]
        ang = [torch.zeros(1, self.hidden_size, device=device)
               for _ in range(self.engine.num_layers)]
        wnd = [torch.zeros(1, self.hidden_size, device=device)
               for _ in range(self.engine.num_layers)]
        
        for ch_id in chars:
            x_emb = self.embedding(
                torch.tensor([[ch_id]], dtype=torch.long, device=device))
            current = x_emb[:, 0, :]
            for i, layer in enumerate(self.engine.layers):
                current, h[i], ang[i], wnd[i] = layer(
                    current, h[i], ang[i], wnd[i])
        
        last_id = chars[-1]
        for _ in range(max_len):
            x_emb = self.embedding(
                torch.tensor([[last_id]], dtype=torch.long, device=device))
            current = x_emb[:, 0, :]
            for i, layer in enumerate(self.engine.layers):
                current, h[i], ang[i], wnd[i] = layer(
                    current, h[i], ang[i], wnd[i])
            
            logits = self.engine.readout(current)[0] / temperature
            if top_k > 0:
                top_vals, top_idx = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float('-inf'))
                mask.scatter_(0, top_idx, top_vals)
                logits = mask
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            if next_id < 10 or next_id >= self.vocab_size:
                next_id = 32
            generated.append(chr(next_id))
            last_id = next_id
        return "".join(generated)


def train_yarnix():
    device = torch.device('cpu')
    print(f"=== YARNIX v7.0 — Multi-Clock Phase ===\n", flush=True)
    print(f"Clock speeds: [0.50 | 0.80 | 0.95 | 0.999]", flush=True)
    print(f"Band layout:  [ultra-fast | fast | slow | ultra-slow]\n", flush=True)
    
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "tinyshakespeare.txt")
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    vocab_size = 128
    chars = [min(ord(c), vocab_size - 1) for c in text]
    data = torch.tensor(chars, dtype=torch.long)
    
    split = int(len(data) * 0.9)
    train_data = data[:split]
    val_data = data[split:]
    print(f"Train: {len(train_data):,} chars | Val: {len(val_data):,} chars",
          flush=True)
    
    seq_len         = 48
    batch_size      = 64
    num_epochs      = 40
    steps_per_epoch = 300
    lr              = 0.001
    warmup_steps    = 300
    
    def get_batch(source):
        ix = torch.randint(len(source) - seq_len - 1, (batch_size,))
        x = torch.stack([source[i : i + seq_len] for i in ix])
        y = torch.stack([source[i+1 : i + seq_len + 1] for i in ix])
        return x.to(device), y.to(device)

    model = YarnixLM(vocab_size=vocab_size, embed_size=64, hidden_size=256).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}\n", flush=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4,
                            betas=(0.9, 0.98))
    total_steps = num_epochs * steps_per_epoch
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            x, y = get_batch(train_data)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        
        avg_train = epoch_loss / steps_per_epoch
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(20):
                x, y = get_batch(val_data)
                logits = model(x)
                vl = criterion(logits.view(-1, vocab_size), y.view(-1))
                val_losses.append(vl.item())
        avg_val = sum(val_losses) / len(val_losses)
        
        elapsed = time.time() - start_time
        lr_now = optimizer.param_groups[0]['lr']
        
        improved = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            improved = " *** NEW BEST ***"
            torch.save(model.state_dict(),
                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "yarnix_best.pt"))
        
        print(f"Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train: {avg_train:.4f} | Val: {avg_val:.4f}{improved} | "
              f"LR: {lr_now:.6f} | {elapsed:.0f}s", flush=True)
        
        if (epoch + 1) % 2 == 0 or epoch == 0:
            sample = model.generate("KING:\n", max_len=150, temperature=0.7)
            clean = "".join([c if c.isprintable() or c == '\n' else '.'
                            for c in sample])
            print(f"--- Sample ---\n{clean}\n--------------\n", flush=True)
    
    print(f"\n=== DONE | Best Val: {best_val_loss:.4f} ===", flush=True)
    for prompt in ["KING:\n", "To be, or ", "The sun ", "O Romeo, "]:
        sample = model.generate(prompt, max_len=250, temperature=0.7)
        clean = "".join([c if c.isprintable() or c == '\n' else '.'
                        for c in sample])
        print(f"\nPrompt: '{prompt.strip()}'")
        print(clean)
        print("-" * 50, flush=True)


if __name__ == '__main__':
    train_yarnix()
