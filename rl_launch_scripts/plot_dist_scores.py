import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

# Configure plotly to use browser
pio.renderers.default = "browser"

# Read the CSV file
df = pd.read_csv('16sep_step_1.csv')


def plot_max_min_gap(df, step: list[int]):
    # Normalize 'step' column to numeric for reliable filtering
    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    df = df[df['step'].isin(step)].copy()
    assert len(df) > 0, f"No data for step(s) {step}"
    # Ensure score is numeric
    df['score'] = pd.to_numeric(df['score'])


    # Compute per-SAE gap directly from raw scores across selected steps
    sae_gap = df.groupby('sae')['score'].agg(lambda s: s.max() - s.min())

    print(f"SAE Max-Min Gaps across steps {step}:")
    print(sae_gap)
    print(f"\nNumber of SAEs: {len(sae_gap)}")
    print("Overall statistics:")
    print(f"Mean: {sae_gap.mean():.4f}")
    print(f"Std: {sae_gap.std():.4f}")
    print(f"Min: {sae_gap.min():.4f}")
    print(f"Max: {sae_gap.max():.4f}")

    # Plot histogram of gaps as percentages
    fig = px.histogram(
        x=sae_gap.values,
        nbins=20,
        title=f'Distribution of SAE Score Gaps (max-min) across steps {step}',
        labels={'x': 'Score Gap (max - min)', 'y': 'Percentage'},
        histnorm='percent',
    )

    fig.update_layout(
        xaxis_title='Score Gap (max - min)',
        yaxis_title='Percentage',
        yaxis=dict(range=[0, 50], ticksuffix='%'),
        showlegend=False
    )

    fig.show()


def plot_max_min_gap_count(df, step: list[int]):
    # Normalize 'step' column to numeric for reliable filtering
    
    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    df = df[df['step'].isin(step)].copy()
    assert len(df) > 0, f"No data for step(s) {step}"
    # Ensure score is numeric
    df['score'] = pd.to_numeric(df['score'])
    # print the first 5 rows
    print(df.head())


    # Compute per-SAE gap directly from raw scores across selected steps
    sae_gap = df.groupby('sae')['score'].agg(lambda s: s.max() - s.min())

    print(f"SAE Max-Min Gaps (Counts) across steps {step}:")
    print(sae_gap)
    print(f"\nNumber of SAEs: {len(sae_gap)}")
    print("Overall statistics:")
    print(f"Mean: {sae_gap.mean():.4f}")
    print(f"Std: {sae_gap.std():.4f}")
    print(f"Min: {sae_gap.min():.4f}")
    print(f"Max: {sae_gap.max():.4f}")

    # Plot histogram of gaps using counts
    fig = px.histogram(
        x=sae_gap.values,
        nbins=10,
        title=f'Distribution of SAE Score Gaps (max-min) across steps {step}',
        labels={'x': 'Score Gap (max - min)', 'y': 'Count'},
    )

    fig.update_layout(
        xaxis_title='Score Gap (max - min)',
        yaxis_title='Count',
        showlegend=False
    )

    fig.show()


def plot_dist_scores(df, step: list[int]):
    # Normalize 'step' column to numeric for reliable filtering
    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    df = df[df['step'].isin(step)].copy()
    assert len(df) > 0, f"No data for step {step}"
    # Convert score column to numeric (in case it's read as string)
    df['score'] = pd.to_numeric(df['score'])

    # Group by SAE and calculate mean scores
    sae_mean_scores = df.groupby('sae')['score'].mean()

    print("SAE Mean Scores:")
    print(sae_mean_scores)
    print(f"\nNumber of SAEs: {len(sae_mean_scores)}")
    print(f"Overall statistics:")
    print(f"Mean: {sae_mean_scores.mean():.4f}")
    print(f"Std: {sae_mean_scores.std():.4f}")
    print(f"Min: {sae_mean_scores.min():.4f}")
    print(f"Max: {sae_mean_scores.max():.4f}")

    # Create histogram of mean scores distribution
    fig = px.histogram(
        x=sae_mean_scores.values,
        nbins=10,
        title='Distribution of Mean Scores by SAE',
        labels={'x': 'Mean Score', 'y': 'Percentage'},
        histnorm='percent',
        # marginal='box'  # Add box plot on top
    )

    fig.update_layout(
        xaxis_title='Mean Score',
        yaxis_title='Percentage',
        yaxis=dict(range=[0, 50], ticksuffix='%'),
        showlegend=False
    )

    # show in browser
    fig.show()



# New: Violin plot of score distribution per SAE for selected steps
def plot_group_dist(df, step: list[int]):
    # Normalize 'step' column to numeric for reliable filtering
    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    df = df[df['step'].isin(step)].copy()
    assert len(df) > 0, f"No data for step(s) {step}"

    # Ensure score is numeric and create categorical SAE label
    df['score'] = pd.to_numeric(df['score'])
    df['sae_str'] = df['sae'].astype(str)

    # Optional: order SAEs by mean score for readability
    sae_order = (
        df.groupby('sae_str')['score']
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Create violin plot per SAE
    fig = px.violin(
        df,
        x='sae_str',
        y='score',
        category_orders={'sae_str': sae_order},
        # box=True,
        title=f'Score Distribution per SAE (steps {step})',
    )

    fig.update_layout(
        xaxis_title='SAE',
        yaxis_title='Score',
        showlegend=False,
    )

    fig.show()

 
# New: Range-only plot (min–max vertical segments) per SAE for selected steps
def plot_group_range(df, step: list[int]):
    # Normalize 'step' column to numeric for reliable filtering
    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    df = df[df['step'].isin(step)].copy()
    assert len(df) > 0, f"No data for step(s) {step}"

    # Ensure score is numeric and create categorical SAE label
    df['score'] = pd.to_numeric(df['score'])
    df['sae_str'] = df['sae'].astype(str)

    # Compute min and max per SAE (order by mean for readability)
    stats = (
        df.groupby('sae_str')['score']
        .agg(['min', 'max', 'mean'])
        .sort_values('mean', ascending=False)
        .reset_index()
    )

    fig = go.Figure()
    for _, row in stats.iterrows():
        sae_label = row['sae_str']
        fig.add_trace(
            go.Scatter(
                x=[sae_label, sae_label],
                y=[row['min'], row['max']],
                mode='lines',
                line=dict(color='royalblue', width=2),
                showlegend=False,
                hoverinfo='x+y',
            )
        )

    # Overlay jittered individual points for visibility
    scatter_fig = px.strip(
        df,
        x='sae_str',
        y='score',
        category_orders={'sae_str': stats['sae_str'].tolist()},
        color_discrete_sequence=['royalblue'],
    )
    for tr in scatter_fig.data:
        tr.showlegend = False
        tr.update(jitter=1)
        tr.marker.update(size=5)
        fig.add_trace(tr)

    fig.update_layout(
        title=f'Score Range (min–max) per SAE (steps {step})',
        xaxis_title='SAE',
        yaxis_title='Score',
    )

    fig.show()

# plot_dist_scores(df, [9, 10, 11])
# plot_max_min_gap(df, [100, 101, 102, 103, 104])
# plot_dist_scores(df, [1])
# plot_max_min_gap_count(df, [100, 101, 102, 103, 104])
# plot_group_dist(df, [1])
plot_group_range(df, [1])
# plot_max_min_gap(df, [1])
# plot_max_min_gap(df, [9, 10, 11])
# plot_max_min_gap(df, [80, 81, 82, 83, 84])