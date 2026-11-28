"""PosPrune Positional Bias measurement for multi-camera SmolVLM."""
from pathlib import Path
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from .cache import CAMERA_ORDER, record_key, records


def sample_disjoint_image_groups(prepared_dir, split='train', samples_per_group=1000, seed=42):
    """
    Sample two disjoint groups of dataset entries such that:
    1. Group 1 and Group 2 share NO images/scenes in common.
    2. Within each group, each sample comes from a distinct image/scene.
    """
    jsonl_path = Path(prepared_dir) / f'{split}.jsonl'
    if not jsonl_path.exists():
        raise FileNotFoundError(f'{jsonl_path} does not exist')

    # Group all record indices by unique image/scene key
    image_to_indices = {}
    for idx, row in enumerate(records(jsonl_path)):
        key = record_key(row)
        if key not in image_to_indices:
            image_to_indices[key] = []
        image_to_indices[key].append(idx)

    unique_keys = list(image_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_keys)

    max_possible = len(unique_keys) // 2
    actual_n = min(samples_per_group, max_possible)
    if actual_n < samples_per_group:
        print(f"[Warning] Only {len(unique_keys)} unique images available in '{split}'. "
              f"Setting group size to {actual_n} (requested {samples_per_group}).")

    keys_g1 = unique_keys[:actual_n]
    keys_g2 = unique_keys[actual_n: 2 * actual_n]

    # Pick 1 sample index per unique image
    indices_g1 = [rng.choice(image_to_indices[k]) for k in keys_g1]
    indices_g2 = [rng.choice(image_to_indices[k]) for k in keys_g2]

    # Sanity check: Ensure strict disjointness
    set_g1 = set(keys_g1)
    set_g2 = set(keys_g2)
    assert len(set_g1.intersection(set_g2)) == 0, "Groups must not share any images!"

    print(f"Sampled 2 disjoint groups: Group 1 ({len(indices_g1)} images), Group 2 ({len(indices_g2)} images).")
    return indices_g1, indices_g2


def compute_group_positional_attention(session, split, sample_indices, target_layers=(2, 18), normalize=True):
    """
    Compute average attention score distribution over the 384 visual tokens
    for the specified target layers across a given group of samples.
    """
    model = session.model
    device = session.device
    processor = session.processor
    image_token_id = processor.tokenizer.convert_tokens_to_ids(str(processor.image_token))

    # Output storage: {layer_idx: np.zeros(384)}
    accumulated_attn = {layer: np.zeros(384, dtype=np.float64) for layer in target_layers}
    
    with torch.inference_mode():
        for idx in tqdm(sample_indices, desc="Measuring Positional Bias"):
            sample = session.cache.sample(split, int(idx))
            item = {**sample, 'visual_features': torch.from_numpy(sample['visual_features'])}
            batch = session.collator([item])

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            visual_features = batch['visual_features'].to(device).float()

            # Forward pre-connector & merger
            projected = model._project(visual_features, input_ids.shape[0])
            embeddings = model.native.get_input_embeddings()(input_ids)
            fused = model.native.model.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=embeddings,
                image_hidden_states=projected
            )

            # Eager forward pass through text decoder
            outputs = model.native.model.text_model(
                inputs_embeds=fused,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True
            )

            # Identify visual token positions (exactly 384 tokens)
            seq_ids = input_ids[0]
            visual_indices = torch.where(seq_ids == image_token_id)[0].cpu()

            # Last instruction token index (query token)
            last_token_idx = attention_mask[0].nonzero()[-1].item()

            for layer in target_layers:
                layer_attn = outputs.attentions[layer][0]  # [num_heads, seq_len, seq_len]
                # Average across heads: [seq_len]
                mean_head_attn = layer_attn[:, last_token_idx, :].mean(dim=0).cpu().numpy()
                v_attn = mean_head_attn[visual_indices]  # [384]

                if normalize:
                    # Normalize over visual tokens so sum = 1.0 per sample
                    s = v_attn.sum()
                    if s > 1e-12:
                        v_attn = v_attn / s

                accumulated_attn[layer] += v_attn

    # Average over group samples
    num_samples = len(sample_indices)
    mean_attn = {layer: accumulated_attn[layer] / num_samples for layer in target_layers}
    return mean_attn


def plot_posprune_figure3(results_g1, results_g2, target_layers=(2, 18),
                         layer_names=('Shallow Layer (Layer 2)', 'Deep Layer (Layer 18)'),
                         save_path='figure3_positional_bias.png'):
    """
    Publication-ready Figure 3 recreation matching PosPrune style & figures4papers aesthetics.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.6,
        'lines.linewidth': 1.8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    COLOR_G1 = '#1f77b4'  # Blue
    COLOR_G2 = '#ff7f0e'  # Orange
    CAMERAS = CAMERA_ORDER

    num_panels = len(target_layers)
    fig, axes = plt.subplots(1, num_panels, figsize=(7.2 * num_panels, 4.5), dpi=300)
    if num_panels == 1:
        axes = [axes]

    x = np.arange(384)

    for idx, (layer, ax) in enumerate(zip(target_layers, axes)):
        curve_g1 = results_g1[layer]
        curve_g2 = results_g2[layer]

        # Calculate correlation between the two independent random groups
        r_val, _ = pearsonr(curve_g1, curve_g2)
        rho_val, _ = spearmanr(curve_g1, curve_g2)

        # Plot curves
        ax.plot(x, curve_g1, color=COLOR_G1, label='Group 1 (1,000 distinct images)', alpha=0.9)
        ax.plot(x, curve_g2, color=COLOR_G2, linestyle='--', label='Group 2 (1,000 distinct images)', alpha=0.9)

        # Camera boundary demarcations
        for c_idx in range(1, 6):
            ax.axvline(c_idx * 64 - 0.5, color='#aaaaaa', linestyle=':', linewidth=0.9, alpha=0.8)

        # Camera labels at bottom/top
        y_max = max(curve_g1.max(), curve_g2.max()) * 1.15
        y_min = min(curve_g1.min(), curve_g2.min()) * 0.85
        ax.set_ylim(y_min, y_max)

        for c_idx, cam in enumerate(CAMERAS):
            # Alternating light background shading for cameras
            if c_idx % 2 == 1:
                ax.axvspan(c_idx * 64, (c_idx + 1) * 64 - 1, color='#f4f4f4', alpha=0.5, zorder=0)
            ax.text(c_idx * 64 + 32, y_max * 0.94, cam, fontsize=8.5, ha='center',
                    color='#444444', weight='semibold')

        # Annotation box for correlation
        title_prefix = f"({chr(97 + idx)}) {layer_names[idx]}"
        ax.set_title(title_prefix, weight='bold', pad=12)
        ax.set_xlabel('Visual Token Position Index')
        ax.set_ylabel('Mean Relative Attention Score')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, linestyle='--', color='#e0e0e0', alpha=0.7, axis='y')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(64))

        # Correlation text box
        corr_text = f"Pearson $r = {r_val:.4f}$\nSpearman $\\rho = {rho_val:.4f}$"
        ax.text(0.04, 0.78, corr_text, transform=ax.transAxes, fontsize=9.5,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))

        ax.legend(loc='upper right', frameon=False)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        pdf_path = str(Path(save_path).with_suffix('.pdf'))
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Figures saved to:\n  - {save_path}\n  - {pdf_path}")
    plt.show()
