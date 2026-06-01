import matplotlib.pyplot as plt


def _save_metric_line_plot(df_sub, metric, ylabel, output_path, title_suffix, noise_types, snr_levels):
    fig, ax = plt.subplots(figsize=(8, 5))

    for noise_type in noise_types:
        subset = df_sub[df_sub["noise_type"] == noise_type].groupby("snr_db")[metric].mean()
        ax.plot(subset.index, subset.values * 100, marker="o", label=noise_type)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric.upper()} vs SNR by Noise Type{title_suffix}")
    ax.set_xticks(snr_levels)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path.name}")
 

def _save_metric_heatmap(df_sub, metric, output_path, title_suffix):
    fig, ax = plt.subplots(figsize=(8, 4))

    pivot = df_sub.groupby(["noise_type", "snr_db"])[metric].mean().unstack()
    heatmap = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Noise Type")
    ax.set_title(f"{metric.upper()} Heatmap (Noise Type x SNR){title_suffix}")

    for row_idx in range(pivot.values.shape[0]):
        for col_idx in range(pivot.values.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{pivot.values[row_idx, col_idx]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    plt.colorbar(heatmap, ax=ax, label=metric.upper())
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path.name}")


def save_plots(df_sub, charts_dir, suffix, title_suffix, noise_types, snr_levels):
    _save_metric_line_plot(
        df_sub=df_sub,
        metric="wer",
        ylabel="WER (%)",
        output_path=charts_dir / f"wer_vs_snr{suffix}.png",
        title_suffix=title_suffix,
        noise_types=noise_types,
        snr_levels=snr_levels,
    )
    _save_metric_line_plot(
        df_sub=df_sub,
        metric="cer",
        ylabel="CER (%)",
        output_path=charts_dir / f"cer_vs_snr{suffix}.png",
        title_suffix=title_suffix,
        noise_types=noise_types,
        snr_levels=snr_levels,
    )
    _save_metric_heatmap(
        df_sub=df_sub,
        metric="wer",
        output_path=charts_dir / f"wer_heatmap{suffix}.png",
        title_suffix=title_suffix,
    )
    _save_metric_heatmap(
        df_sub=df_sub,
        metric="cer",
        output_path=charts_dir / f"cer_heatmap{suffix}.png",
        title_suffix=title_suffix,
    )


def save_all_plots(df, selected_voices, charts_dir, noise_types, snr_levels):
    save_plots(df, charts_dir, "", "", noise_types, snr_levels)

    for voice_name in selected_voices:
        df_voice = df[df["voice_name"] == voice_name]
        safe_name = voice_name.replace("/", "_")
        save_plots(
            df_voice,
            charts_dir,
            f"_{safe_name}",
            f" - {voice_name}",
            noise_types,
            snr_levels,
        )
