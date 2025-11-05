import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import numpy as np


def plot_juror_histograms_for_stage_korekta(stage_name, stage_punkty_df, stage_korekta_df):
    """
    Calculates and displays histograms of juror votes for a given stage.
    """
    print(f"\n--- Juror score histograms for stage: {stage_name} {" (with correction)" if stage_korekta_df is not None else " (without correction)"} ---")

    juror_columns = [col for col in stage_punkty_df.columns[4:-1] if col != 'average score']
    juror_stats = stage_punkty_df[juror_columns].agg(['mean']).T
    juror_stats = juror_stats.sort_values(by='mean').reset_index()
    juror_stats.rename(columns={'index': 'juror'}, inplace=True)

    sorted_jurors = juror_stats['juror'].tolist()

    n_rows = 9
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, n_rows * 2))
    fig.suptitle(f'Juror score histograms for stage: {stage_name} {" (with correction)" if stage_korekta_df is not None else " (without correction)"}', fontsize=16)

    for i in range(n_rows):
        # Plot for the first column
        if i < len(sorted_jurors):
            juror1 = sorted_jurors[i]
            plot_chart_with_corrections(axes, i, 0, juror1, stage_korekta_df, stage_punkty_df)

        # Plot for the second column
        juror2_index = i + 9
        if juror2_index < len(sorted_jurors):
            juror2 = sorted_jurors[juror2_index]
            plot_chart_with_corrections(axes, i, 1, juror2, stage_korekta_df, stage_punkty_df)
        else:
            # If there is no juror for the second column, hide the subplot
            if i < n_rows:
                fig.delaxes(axes[i, 1])


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(f'histograms/stage_{stage_name}_histograms{"_korekta" if stage_korekta_df is not None else ""}.png')
    plt.close(fig)


def plot_chart_with_corrections(axes, i, col, juror, stage_korekta_df, stage_punkty_df):
    color_punkty = 'steelblue'
    color_korekta = 'navajowhite'

    scores1_punkty = stage_punkty_df[juror].dropna()
    if stage_korekta_df is not None:
        scores1_korekta = stage_korekta_df[juror].dropna()
    mean_score1_punkty = scores1_punkty.mean()
    if stage_korekta_df is not None:
        mean_score1_korekta = scores1_korekta.mean()
    ax1 = axes[i, col]
    ax1.hist(scores1_punkty, bins=range(10, 27), edgecolor='black', align='left', alpha=0.5, color=color_punkty,label='Points')
    if stage_korekta_df is not None:
        ax1.hist(scores1_korekta.round(), bins=range(10, 27), edgecolor='black', align='left', alpha=0.5, color=color_korekta, label='Correction')
    ax1.axvline(mean_score1_punkty, color=color_punkty, linestyle='--', linewidth=2, label=f'Avg: {mean_score1_punkty:.2f}')
    if stage_korekta_df is not None:
        ax1.axvline(mean_score1_korekta, color=color_korekta, linestyle='--', linewidth=2, label=f'Avg.corr.: {mean_score1_korekta:.2f}')
    ax1.legend()
    ax1.set_title(juror)

def create_single_table(stage_punkty_df, stage_korekta_df, stage_name):
    """
    Analyzes data for a single stage and prints a formatted analysis table.
    """
    print(f"\n\n--- Juror correction analysis for stage: {stage_name} ---")

    if stage_korekta_df.empty:
        print("No data for this stage.")
        return

    juror_columns = [col for col in stage_punkty_df.columns[4:-1] if col != 'average score']
    final_score_col = 'average score'
    juror_stats = []

    for juror in juror_columns:
        punkty_scores = stage_punkty_df[juror].fillna(-1)
        korekta_scores = stage_korekta_df[juror].fillna(-2)
        is_corrected = (punkty_scores != korekta_scores)

        total_corrections = is_corrected.sum()

        is_lower = (stage_korekta_df[juror] < stage_korekta_df[final_score_col])
        is_higher = (stage_korekta_df[juror] > stage_korekta_df[final_score_col])

        corrected_lower_count = (is_corrected & is_lower).sum()
        corrected_higher_count = (is_corrected & is_higher).sum()

        juror_stats.append({
            'juror': juror,
            'total_corrections': total_corrections,
            'lower': corrected_lower_count,
            'higher': corrected_higher_count
        })

    sorted_stats = sorted(juror_stats, key=lambda item: item['total_corrections'], reverse=True)
    total_scores_in_stage = len(stage_korekta_df)

    print(f"(Number of participants in the stage: {total_scores_in_stage})")
    for stats in sorted_stats:
        lower_perc = (stats['lower'] / total_scores_in_stage) * 100 if total_scores_in_stage > 0 else 0
        higher_perc = (stats['higher'] / total_scores_in_stage) * 100 if total_scores_in_stage > 0 else 0
        all_perc = (stats['total_corrections'] / total_scores_in_stage) * 100 if total_scores_in_stage > 0 else 0
        print(f"- {stats['juror']}: {stats['total_corrections']} ({all_perc:.2f}%) too low: {stats['lower']} ({lower_perc:.2f}%), too high: {stats['higher']} ({higher_perc:.2f}%)")


def generate_analysis_for_stages(df_punkty, df_korekta, stages_to_analyze):
    """
    Iterates through a list of stages and calls the analysis function for each.
    First, it generates a summary table for all stages combined.
    """
    # Generate summary table for all stages first
    create_single_table(df_punkty, df_korekta, "All stages")

    # Then, generate a table for each individual stage
    for stage in stages_to_analyze:
        stage_punkty_df = df_punkty[df_punkty['stage'] == stage].reset_index(drop=True)
        stage_korekta_df = df_korekta[df_korekta['stage'] == stage].reset_index(drop=True)
        create_single_table(stage_punkty_df, stage_korekta_df, stage)


def print_juror_stats_for_stage(stage_name, stage_punkty_df):
    """
    Calculates and displays statistics for each juror's score in a given stage.
    """
    print(f"\n--- Juror score statistics for stage: {stage_name} (without correction) ---")

    juror_columns = [col for col in stage_punkty_df.columns[4:-1] if col != 'average score']
    
    juror_stats = stage_punkty_df[juror_columns].agg(['min', 'max', 'mean', 'std']).T
    juror_stats = juror_stats.sort_values(by='mean').reset_index()
    juror_stats.rename(columns={'index': 'juror'}, inplace=True)

    print(f"{ 'Juror':<20} {'Min':<5} {'Max':<5} {'Average':<7} {'Std Dev':<7}")
    print("-" * 50)

    for index, row in juror_stats.iterrows():
        print(f"{row['juror']:<20} {row['min']:<5.2f} {row['max']:<5.2f} {row['mean']:<7.2f} {row['std']:<7.2f}")


def print_stage_ranking(stage_name, stage_punkty_df, stage_korekta_df, num_to_promote):
    """
    Prints a formatted ranking for a single stage, showing the effect of score corrections.
    """
    print(f"\n--- Stage Ranking {stage_name} (scores without correction) ---")

    stage_punkty = stage_punkty_df.sort_values(by='stage score', ascending=False).reset_index(drop=True)
    stage_korekta = stage_korekta_df.sort_values(by='stage score', ascending=False).reset_index(drop=True)

    stage_punkty['rank_punkty'] = stage_punkty.index + 1
    stage_korekta['rank_korekta'] = stage_korekta.index + 1

    top_promoted_korekta_names = set(stage_korekta.head(num_to_promote)['Name and Surname'])

    korekta_rank_map = stage_korekta.set_index('Name and Surname')['rank_korekta'].to_dict()
    korekta_score_map = stage_korekta.set_index('Name and Surname')['stage score'].to_dict()

    for i in range(len(stage_punkty)):
        rank_punkty = i + 1
        name_punkty = stage_punkty.loc[i, 'Name and Surname']
        score_punkty = stage_punkty.loc[i, 'stage score']
        score_korekta = korekta_score_map.get(name_punkty, 0.0)
        
        rank_in_korekta = korekta_rank_map.get(name_punkty, rank_punkty)
        rank_change =  rank_in_korekta - rank_punkty
        
        marker = ""
        if i < num_to_promote and name_punkty not in top_promoted_korekta_names:
            marker = " (PROMOTION!)"
        elif i >= num_to_promote and name_punkty in top_promoted_korekta_names:
            marker = " (NO PROMOTION!)"

        print(f"{rank_punkty}. {name_punkty} ({score_korekta:.2f})->({score_punkty:.2f}) ({rank_change:+}){marker}")
        
        if i == num_to_promote - 1:
            print("-" * 80)


def calculate_stage_score(dataframe):
    dataframe['stage score'] = 0.0
    dataframe['stage'] = dataframe['stage'].astype(str)
    participant_scores = dataframe.pivot_table(index='Name and Surname', columns='stage', values='average score').to_dict('index')

    def get_score(participant, stage, p_scores):
        try:
            return p_scores[participant][stage]
        except KeyError:
            return 0.0

    for index, row in dataframe.iterrows():
        participant_id = row['Name and Surname']
        stage = row['stage']

        score = 0.0
        if stage == '1':
            score = get_score(participant_id, '1', participant_scores)
        elif stage == '2':
            score = get_score(participant_id, '1', participant_scores) * 0.3 + get_score(participant_id, '2', participant_scores) * 0.7
        elif stage == '3':
            score = get_score(participant_id, '1', participant_scores) * 0.1 + get_score(participant_id, '2', participant_scores) * 0.2 + get_score(participant_id, '3', participant_scores) * 0.7
        elif stage == 'final':
            score = get_score(participant_id, '1', participant_scores) * 0.1 + get_score(participant_id, '2', participant_scores) * 0.2 + get_score(participant_id, '3', participant_scores) * 0.35 + get_score(participant_id, 'final', participant_scores) * 0.35

        dataframe.loc[index, 'stage score'] = score
    return dataframe


def calculate_juror_correlation_matrix(df_dane, stages_to_analyze):
    """
    Calculates and returns the correlation matrix of scores between jurors for the given stages.
    """
    if stages_to_analyze:
        df_dane = df_dane[df_dane['stage'].isin(stages_to_analyze)]

    juror_columns = [col for col in df_dane.columns if col not in ['Nr', 'Name and Surname', 'type', 'stage', 'average score', 'stage score']]
    correlation_matrix = df_dane[juror_columns].corr()

    # Calculate the mean correlation for each juror, excluding the diagonal
    corr_copy = correlation_matrix.copy()
    for col in corr_copy.columns:
        corr_copy.loc[col, col] = None
    mean_corr = corr_copy.mean()
    correlation_matrix.loc['Average'] = mean_corr

    return correlation_matrix


def style_correlation_matrix(corr_matrix):
    """
    Applies background gradient styling to a correlation matrix.
    """
    s = corr_matrix.style.background_gradient(cmap='RdYlGn_r', vmin=0, vmax=1).format("{:.3f}")
    s.set_table_styles([{'selector': '', 'props': [('font-family', 'sans-serif')]},
                        {'selector': 'h1', 'props': [('font-family', 'sans-serif')]},
                        {'selector': 'p', 'props': [('font-family', 'sans-serif')]},
                        {'selector': 'td', 'props': [('text-align', 'right'), ('white-space', 'nowrap')]},
                        {'selector': 'th.row_heading', 'props': [('text-align', 'left'), ('white-space', 'nowrap'), ('min-width', '150px')]},
                        {'selector': 'th.col_heading', 'props': [('text-align', 'right')]},
                        {'selector': 'tr:last-child', 'props': [('height', '40px'), ('font-weight',  'bold'), ('color', 'black')]}])
    return s.to_html()


def perform_cluster_analysis(corr_matrix, stages, correlation_threshold):
    """
    Performs cluster analysis on the correlation matrix and identifies groups of jurors.
    """
    distance_matrix = 1 - corr_matrix
    condensed_distance_matrix = squareform(distance_matrix)
    linked = hierarchy.linkage(condensed_distance_matrix, 'average')

    plt.figure(figsize=(10, 7))
    dendrogram = hierarchy.dendrogram(linked,
                                      orientation='top',
                                      labels=corr_matrix.index,
                                      distance_sort='descending',
                                      show_leaf_counts=True)
    plt.title(f'Juror Clustering For Stage {stages}')
    plt.xlabel('Juror')
    plt.ylabel('Correlation')
    ax = plt.gca()
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{1 - x:.2f}' for x in yticks])
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    stage_str = "_".join(stages)
    plt.savefig(f'img/dendrogram_{stage_str}.png')
    plt.close()

    distance_threshold = 1 - correlation_threshold
    clusters = hierarchy.fcluster(linked, distance_threshold, criterion='distance')

    # Group jurors by cluster
    grouped_jurors = {}
    clusters_list = clusters.tolist()
    for juror, cluster_id in zip(corr_matrix.index, clusters_list):
        if cluster_id not in grouped_jurors:
            grouped_jurors[cluster_id] = []
        grouped_jurors[cluster_id].append(juror)

    print(f"\nJuror groups for stage {", ".join(stages)} at correlation threshold: {correlation_threshold}")
    for cluster_id, jurors in sorted(grouped_jurors.items()):
        print(f"  Group {cluster_id}: {", ".join(jurors)}")

def analyze_and_visualize(df_punkty, df_korekta):
    """Analyzes and visualizes the data."""

    stages = sorted(df_punkty['stage'].unique())

    promotion_config = {
        '1': 40,
        '2': 20,
        '3': 11,
        'final' : 11
    }

    for stage in stages:
        stage_punkty_df = df_punkty[df_punkty['stage'] == stage]
        stage_korekta_df = df_korekta[df_korekta['stage'] == stage]
        print_juror_stats_for_stage(stage, stage_punkty_df)
        plot_juror_histograms_for_stage_korekta(stage, stage_punkty_df, None)
        plot_juror_histograms_for_stage_korekta(stage, stage_punkty_df, stage_korekta_df)
    plot_juror_histograms_for_stage_korekta("ALL", df_punkty, None)
    plot_juror_histograms_for_stage_korekta("ALL", df_punkty, df_korekta)

    for stage in stages:
        if stage in promotion_config:
            stage_punkty_df = df_punkty[df_punkty['stage'] == stage]
            stage_korekta_df = df_korekta[df_korekta['stage'] == stage]

            num_to_promote = promotion_config[stage]
            print_stage_ranking(stage, stage_punkty_df, stage_korekta_df, num_to_promote)

    df_punkty_analysis = df_punkty.copy().reset_index(drop=True)
    df_korekta_analysis = df_korekta.copy().reset_index(drop=True)

    stages = sorted(df_korekta_analysis['stage'].unique())
    generate_analysis_for_stages(df_punkty_analysis, df_korekta_analysis, stages)

    with open("html/corr_punkty.html", "w", encoding='utf-8') as f:
        f.write("<h1>Juror Correlation Matrix (before correction)</h1>")

    correlation_calc_and_print(df_punkty, ['1', '2', '3', 'final'], .3)
    correlation_calc_and_print(df_punkty, ['1'], .4)
    correlation_calc_and_print(df_punkty, ['2'], .3)
    correlation_calc_and_print(df_punkty, ['3'], .3)
    correlation_calc_and_print(df_punkty, ['final'], .2)

def correlation_calc_and_print(df_punkty, stages_for_correlation, correlation_threshold):
    corr_punkty = calculate_juror_correlation_matrix(df_punkty, stages_for_correlation)
    toHtmlFile(corr_punkty, stages_for_correlation)

    corr_punkty_for_clustering = corr_punkty.drop('Average')
    perform_cluster_analysis(corr_punkty_for_clustering, stages_for_correlation,  correlation_threshold)


def toHtmlFile(corr_punkty, stages):
    styled_corr_punkty = style_correlation_matrix(corr_punkty)
    html_output = f"""
<html><head><meta charset="UTF-8"></head>
<body>
<p>Stage: {", ".join(stages)}</p>
{styled_corr_punkty}
</body></html>'
"""
    with open("html/corr_punkty.html", "a+", encoding='utf-8') as f:
        f.write(html_output)


def main():
    """Main function to load, clean, and analyze data."""
    try:
        df = pd.read_csv('chopin_competition_scores.csv', delimiter=',', na_values=0.0)
        df.columns = df.columns.str.strip()
        df.rename(columns={'Imię i Nazwisko': 'Name and Surname', 'typ': 'type', 'średnia ocena': 'average score', 'wynik etapu': 'stage score'}, inplace=True)
        df['type'] = df['type'].replace({'punkty': 'points', 'korekta': 'correction'})
        score_columns = df.columns[4:]
        for col in score_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df_punkty = calculate_stage_score(df[df['type'] == 'points'].copy())
        df_korekta = calculate_stage_score(df[df['type'] == 'correction'].copy())

        analyze_and_visualize(df_punkty, df_korekta)

    except FileNotFoundError as e:
        print(f"Error: File not found: {e.filename}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()