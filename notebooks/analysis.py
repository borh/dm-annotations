import marimo

__generated_with = "0.15.2"
app = marimo.App(width="columns", app_title="学術論文のセクション分類及び接続表現と文末表現の使用傾向")

with app.setup(hide_code=True):
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from dm_annotations.io.text_loader import (
        list_folder_sources,
        parse_md_to_section_nodes,
        render_md_sections_html,
    )
    from dm_annotations.notebook import (
        build_contingency_table,
        build_doc_minimap_for_file,
        chi2_standardized_residuals,
        collocation_sentences_html,
        compute_pmi_entropy,
        count_dms_df,
        expand_function_columns,
        extract_dms_df,
        load_sections_dataframe,
        sample_by_section,
        visualize_ca,
        llr_keyness,
    )


@app.cell(hide_code=True)
def _():
    def selected_genre_col(
        df: pl.DataFrame, preferred: str | None, fallback: str = "ジャンル"
    ) -> str | None:
        if preferred and preferred in df.columns:
            return preferred
        return fallback if fallback in df.columns else None

    def add_selected_genre(df: pl.DataFrame, preferred: str | None) -> pl.DataFrame:
        col = selected_genre_col(df, preferred)
        return (
            df.with_columns(pl.col(col).alias("_selected_genre"))
            if col
            else df.with_columns(pl.lit(None).alias("_selected_genre"))
        )

    def filter_types(
        df: pl.DataFrame, include_conn: bool, include_end: bool
    ) -> pl.DataFrame:
        kinds: list[str] = []
        if include_conn:
            kinds.append("接続表現")
        if include_end:
            kinds.append("文末表現")
        return df if not kinds else df.filter(pl.col("タイプ").is_in(kinds))

    def ensure_function_columns(df: pl.DataFrame) -> pl.DataFrame:
        return df if "機能_1" in df.columns else expand_function_columns(df)

    def create_aggregated_df(df: pl.DataFrame, agg_col: str) -> pl.DataFrame:
        dfx = ensure_function_columns(df)
        if agg_col not in dfx.columns:
            agg_col = "表現"
        base = [
            pl.col(agg_col).alias("表現"),
            pl.col("タイプ"),
            pl.col("section"),
            pl.col("title"),
            pl.col("sentence_id"),
            pl.col("position"),
            pl.col("span_text"),
        ]
        opt = [pl.col(c) for c in ["genre1", "genre2", "genre3"] if c in dfx.columns]
        return dfx.select(base + opt)

    def group_cols_for(mode: str) -> list[str] | None:
        if mode == "none":
            return None
        if mode == "section":
            return ["section"]
        if mode == "genre":
            return ["_selected_genre"]
        return ["section", "_selected_genre"]

    def pmi_sorted(df: pl.DataFrame, group_cols: list[str] | None) -> pl.DataFrame:
        return compute_pmi_entropy(df, group_cols=group_cols).sort(
            "pmi", descending=True
        )
    return (
        add_selected_genre,
        create_aggregated_df,
        filter_types,
        group_cols_for,
        pmi_sorted,
    )


@app.cell
def _(folder):
    mo.md(
        f"""
    # コーパス情報

    コーパスファイル：{len(list(folder.glob("*.md")))}
    """
    )
    return


@app.cell
def _():
    CE_LEVELS = ["表現", "機能_1", "機能_2", "機能_3"]
    SFE_LEVELS = ["表現", "機能_1", "機能_2"]
    return CE_LEVELS, SFE_LEVELS


@app.cell
def _(CE_LEVELS, SFE_LEVELS, df_fx):
    conn_cols = [c for c in CE_LEVELS if c in df_fx.columns]
    end_cols = [c for c in SFE_LEVELS if c in df_fx.columns]

    conn_catalog = (
        df_fx.filter(pl.col("タイプ") == "接続表現")
        .select(conn_cols)
        .unique()
        .sort("表現")
    )
    end_catalog = (
        df_fx.filter(pl.col("タイプ") == "文末表現")
        .select(end_cols)
        .unique()
        .sort("表現")
    )

    mo.vstack(
        [
            mo.md("### 表現カタログ（接続表現）"),
            mo.ui.table(conn_catalog, selection=None),
            mo.md("### 表現カタログ（文末表現）"),
            mo.ui.table(end_catalog, selection=None),
        ]
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    ## 節処理の確認

    節の分類が正しいかどうかをファイルごとに確認できる。
    """
    )
    return


@app.cell
def _(basename_to_record, file_selector):
    selected_file_record = basename_to_record(file_selector.value)
    selected_file_record
    return (selected_file_record,)


@app.cell
def _(folder):
    files_df = list_folder_sources(folder)

    def basename_to_record(bn: str):
        record = files_df.filter(pl.col("basename") == bn).row(0, named=True)
        return record

    file_selector = mo.ui.dropdown(
        options=files_df["basename"],
        value="00003",
        label="テキストの節処理確認",
        searchable=True,
    )
    file_selector
    return basename_to_record, file_selector


@app.cell
def _(selected_file_record):
    p = (
        Path(selected_file_record["path"]) 
        if selected_file_record.get("path")
        else Path(selected_file_record["preprocessed_path"])
    )
    segment_chars = 100
    interactive_map = build_doc_minimap_for_file(
        p, segment_chars=segment_chars, wrap_cols=20, interactive=True
    )
    mo.vstack(
        [
            mo.md("### Document mini map"),
            interactive_map,
        ]
    )
    return


@app.cell
def _(selected_file_record):
    mo.Html(render_md_sections_html(Path(selected_file_record["path"])))
    return


@app.cell
def _(selected_file_record):
    _output = None
    x = parse_md_to_section_nodes(Path(selected_file_record["preprocessed_path"]))
    _output = x
    _output
    return


@app.cell
def _(selected_file_record):
    mo.md("```\n" + Path(selected_file_record["preprocessed_path"]).read_text() + "\n```")
    return


@app.cell
def _():
    mo.md(
        r"""
    ## 接続・文末表現の節分類
    ### 設定・データ
    """
    )
    return


@app.cell
def _():
    folder = Path("./final-conversion-sonnet-4")
    return (folder,)


@app.cell
def _(folder):
    # with mo.persistent_cache("df_sections_5"):
    df_sections = load_sections_dataframe(folder, ext=None)
    core_sections_defaults = [
        "introduction",
        "background",
        "literature_review",
        "methods",
        # "data",
        "results",
        "discussion",
        "conclusion",
        # "future_work",
    ]
    return core_sections_defaults, df_sections


@app.cell
def _(core_sections_defaults, df_sections):
    # Determine available sections (works for LazyFrame or DataFrame)
    if isinstance(df_sections, pl.LazyFrame):
        sections_available = (
            df_sections.select("section").unique().collect()["section"].to_list()
        )
    else:
        sections_available = df_sections.select("section").unique()["section"].to_list()

    default_selected = [s for s in core_sections_defaults if s in sections_available]
    if not default_selected:
        default_selected = sections_available

    section_selector = mo.ui.multiselect(
        options=sorted(sections_available),
        value=sorted(default_selected),
        label="Sections to include",
    )
    mo.vstack([mo.md("対象セクションの選択"), section_selector])
    return section_selector, sections_available


@app.cell
def _(df_sections, section_selector, sections_available):
    allowed = section_selector.value or sections_available
    filtered_sections_df = df_sections.filter(pl.col("section").is_in(allowed))
    return (filtered_sections_df,)


@app.cell
def _(filtered_sections_df):
    # Section counts for selected sections, apply k-threshold
    section_counts = (
        filtered_sections_df.group_by("section")
        .agg(pl.len().alias("count"))
        # .filter(pl.col("count") >= min_section_count.value)
    ).collect()

    valid_sections = set(section_counts["section"].to_list())

    # Choose which genre level to plot
    cols = (
        filtered_sections_df.columns if hasattr(filtered_sections_df, "columns") else []
    )
    genre_cols = [c for c in ["genre", "genre1", "genre2", "genre3"] if c in cols]
    gcol = mo.ui.dropdown(
        options=genre_cols if genre_cols else ["genre"],
        value=(genre_cols[2] if genre_cols else "genre2"),
        label="Genre level for plots (sections)",
        searchable=True,
    )
    mo.vstack([gcol])
    return gcol, section_counts, valid_sections


@app.cell
def _(filtered_sections_df, gcol, section_counts):
    df_sel = (
        filtered_sections_df.collect()
        if isinstance(filtered_sections_df, pl.LazyFrame)
        else filtered_sections_df
    )

    # Genre distribution over the currently selected sections
    s_genre_counts = (
        df_sel.filter(pl.col(gcol.value).is_not_null())
        .group_by(gcol.value)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    # Altair charts
    section_chart = (
        alt.Chart(section_counts)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Rows"),
            y=alt.Y("section:N", sort="-x", title="Section"),
            tooltip=[alt.Tooltip("section:N"), alt.Tooltip("count:Q", title="rows")],
        )
        .properties(width=520, height=min(600, 24 * max(1, len(section_counts))))
    )

    genre_chart = (
        alt.Chart(s_genre_counts)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Rows"),
            y=alt.Y(f"{gcol.value}:N", sort="-x", title=gcol.value),
            color=alt.Color(f"{gcol.value}:N", legend=None),
            tooltip=[
                alt.Tooltip(f"{gcol.value}:N", title=gcol.value),
                alt.Tooltip("count:Q", title="rows"),
            ],
        )
        .properties(width=520, height=min(600, 24 * max(1, len(s_genre_counts))))
    )

    section_count_display = section_counts.sort("count", descending=True)

    mo.vstack(
        [
            mo.md("### Section counts (before filtering categories)"),
            mo.vstack(
                [
                    mo.ui.table(section_count_display, label=None, selection=None),
                    section_chart,
                ]
            ),
            mo.md("### Genre distribution (selected sections)"),
            mo.vstack(
                [
                    mo.ui.table(s_genre_counts, label=None, selection=None),
                    genre_chart,
                ]
            ),
        ]
    )
    return


@app.cell
def _():
    max_samples_per_section = mo.ui.slider(
        label="Maximum samples per section (downsampling)",
        start=10,
        stop=5000,
        value=5000,
        step=10,
        show_value=True,
        include_input=True,
    )
    balance_sampling_toggle = mo.ui.switch(True, label="Balance sampling by genre2")
    mo.hstack([max_samples_per_section, balance_sampling_toggle])
    return balance_sampling_toggle, max_samples_per_section


@app.cell
def _(
    balance_sampling_toggle,
    filtered_sections_df,
    max_samples_per_section,
    valid_sections,
):
    base_for_sampling = (
        filtered_sections_df.filter(pl.col("section").is_in(list(valid_sections)))
        if valid_sections
        else filtered_sections_df
    )
    sampled_sections_df = sample_by_section(
        base_for_sampling,
        max_rows_per_section=max_samples_per_section.value,
        section_col="section",
        balance_on="genre2" if balance_sampling_toggle.value else None,
        seed=42,
    )
    sampled_sections_df
    return (sampled_sections_df,)


@app.cell
def _(sampled_sections_df):
    sampled_section_counts = (
        sampled_sections_df.group_by("section")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    mo.ui.table(sampled_section_counts, label="最終分析対象節分類内訳", selection=None)
    return (sampled_section_counts,)


@app.cell
def _(sampled_section_counts):
    mo.md(f"""セクション合計: {sampled_section_counts.sum()[0,1]}""")
    return


@app.cell(column=1)
def _():
    mo.md(
        """
    # 結果

    ## 共起抽出
    """
    )
    return


@app.cell
def _(sampled_sections_df):
    # with mo.persistent_cache("dms_core_sections_v4"):
    # 共起抽出：
    src_sections = sampled_sections_df
    df_dm_enriched = extract_dms_df(
        src_sections,
        pipe_batch_size=400,
    )
    return (df_dm_enriched,)


@app.cell
def _(df_dm_enriched):
    counts = count_dms_df(df_dm_enriched, ["タイプ", "表現"])
    top = counts.sort("頻度", descending=True)
    mo.vstack(
        [
            mo.ui.table(top, label="Top expressions (overall)", selection=None),
        ]
    )
    return


@app.cell
def _(df_for_analysis, stat_category_selector):
    # Build target selector for Keyness from current category axis
    keyness_cat_col = stat_category_selector.value
    options = (
        df_for_analysis.select(pl.col(keyness_cat_col)).drop_nulls().unique().to_series().to_list()
        if keyness_cat_col in df_for_analysis.columns
        else []
    )
    keyness_target = mo.ui.dropdown(
        options=options or ["(none)"],
        value=(options[0] if options else "(none)"),
        label="Keyness: target category",
        searchable=True,
    )
    keyness_target
    return (keyness_target,)


@app.cell
def _(
    df_for_analysis,
    keyness_correction,
    keyness_min_count,
    keyness_target,
    keyness_top_n,
    stat_category_selector,
):
    # Top 5 expressions per section table
    sections_of_interest = [
        "introduction",
        "literature_review",
        "background",
        "methods",
        "results",
        "discussion",
        "conclusion",
    ]

    # Filter to sections available in the data
    available_sections = (
        df_for_analysis.select("section").unique().to_series().to_list()
    )
    sections_to_process = [s for s in sections_of_interest if s in available_sections]

    top_connectives_by_section = {}
    top_sfes_by_section = {}

    df_connectives = df_for_analysis.filter(pl.col("タイプ") == "接続表現")
    df_sfes = df_for_analysis.filter(pl.col("タイプ") == "文末表現")

    for section_name in sections_to_process:
        # Process connectives
        if not df_connectives.is_empty():
            keyness_conn = llr_keyness(
                df_connectives,
                target=section_name,
                category_col="section",
                expr_col="表現",
                min_count=1,
                correction=None,
            )
            if not keyness_conn.is_empty():
                top_5_conn = (
                    keyness_conn.filter(pl.col("target_rel") > 0)
                    .sort("llr", descending=True)
                    .head(5)
                )
                formatted_conn = top_5_conn.with_columns(
                    (
                        pl.col("表現")
                        + " ("
                        + pl.col("llr").round(1).cast(pl.Utf8)
                        + ")"
                    ).alias("formatted")
                )["formatted"].to_list()
                top_connectives_by_section[section_name] = formatted_conn

        # Process sentence-final expressions
        if not df_sfes.is_empty():
            keyness_sfe = llr_keyness(
                df_sfes,
                target=section_name,
                category_col="section",
                expr_col="表現",
                min_count=1,
                correction=None,
            )
            if not keyness_sfe.is_empty():
                top_5_sfe = (
                    keyness_sfe.filter(pl.col("target_rel") > 0)
                    .sort("llr", descending=True)
                    .head(5)
                )
                formatted_sfe = top_5_sfe.with_columns(
                    (
                        pl.col("表現")
                        + " ("
                        + pl.col("llr").round(1).cast(pl.Utf8)
                        + ")"
                    ).alias("formatted")
                )["formatted"].to_list()
                top_sfes_by_section[section_name] = formatted_sfe

    # Create the summary tables
    rank_col = pl.Series("rank", range(1, 6))
    conn_table_df = pl.DataFrame([rank_col])
    sfe_table_df = pl.DataFrame([rank_col])

    # Use sections_to_process to maintain the desired column order
    for section_name in sections_to_process:
        conn_list = top_connectives_by_section.get(section_name, [])
        conn_list.extend([""] * (5 - len(conn_list)))
        conn_table_df = conn_table_df.with_columns(pl.Series(section_name, conn_list))

        sfe_list = top_sfes_by_section.get(section_name, [])
        sfe_list.extend([""] * (5 - len(sfe_list)))
        sfe_table_df = sfe_table_df.with_columns(pl.Series(section_name, sfe_list))

    keyness_summary_tables = mo.vstack(
        [
            mo.md("### Top 5 Characteristic Connectives by Section (LLR)"),
            mo.ui.table(conn_table_df, selection=None),
            mo.md("### Top 5 Characteristic Sentence-Final Expressions by Section (LLR)"),
            mo.ui.table(sfe_table_df, selection=None),
        ]
    )

    # Keyness based on dropdown selector
    _keyness_out_existing = None

    if keyness_target.value in (None, "(none)"):
        _keyness_out_existing = mo.md("No target selected for Keyness.")
    else:
        out = llr_keyness(
            df_for_analysis,
            target=keyness_target.value,
            category_col=stat_category_selector.value,
            expr_col="表現",
            min_count=keyness_min_count.value,
            correction=keyness_correction.value,
        )
        if out.is_empty():
            _keyness_out_existing = mo.md(
                "No key expressions found for the selected target."
            )
        else:
            over = (
                out.filter(pl.col("target_rel") > 0)
                .sort(["llr", "target_rel"], descending=[True, True])
                .head(keyness_top_n.value)
            )
            under = (
                out.filter(pl.col("target_rel") < 0)
                .sort(["llr", "target_rel"], descending=[True, False])
                .head(keyness_top_n.value)
            )

            tooltip_cols = [
                "表現",
                alt.Tooltip("llr:Q", title="LLR", format=".2f"),
                alt.Tooltip("p_value:Q", title="p-value", format=".3g"),
            ]
            if "p_adj" in out.columns:
                tooltip_cols.append(
                    alt.Tooltip("p_adj:Q", title="p-adj", format=".3g")
                )
            tooltip_cols.extend(
                [
                    alt.Tooltip("a:Q", title="target count"),
                    alt.Tooltip("b:Q", title="rest count"),
                    alt.Tooltip("target_rel:Q", title="Δ rate", format=".4f"),
                ]
            )

            chart_over = (
                alt.Chart(over)
                .mark_bar()
                .encode(
                    x=alt.X("llr:Q", title="LLR"),
                    y=alt.Y("表現:N", sort="-x", title="Expression"),
                    tooltip=tooltip_cols,
                )
                .properties(width=700, height=28 * max(1, len(over)))
            )

            _keyness_out_existing = mo.vstack(
                [
                    mo.md(f"### キーネス（対象: {keyness_target.value}）"),
                    mo.hstack(
                        [
                            mo.vstack(
                                [
                                    mo.md("#### Over-used (target > rest)"),
                                    mo.ui.table(over, selection=None),
                                ]
                            ),
                            mo.vstack(
                                [
                                    mo.md("#### Under-used (target < rest)"),
                                    mo.ui.table(under, selection=None),
                                ]
                            ),
                        ]
                    ),
                    mo.md("#### Over-used (Top-N by LLR)"),
                    chart_over,
                ]
            )

    # Combine new summary tables with the existing output
    _keyness_out = mo.vstack([keyness_summary_tables, _keyness_out_existing])

    _keyness_out
    return


@app.cell
def _(df_dm_enriched, genre_level_selector):
    g_sel = genre_level_selector.value
    if g_sel and g_sel in df_dm_enriched.columns and g_sel != "(none)":
        values = df_dm_enriched.select(pl.col(g_sel)).drop_nulls().unique().to_series().to_list()
        genre_values = mo.ui.multiselect(
            options=sorted(values),
            value=[],
            label=f"Filter values for {g_sel}",
        )
    else:
        genre_values = mo.ui.multiselect(
            options=[],
            value=[],
            label="Filter values (select a genre level)",
        )
    genre_values
    return (genre_values,)


@app.cell
def _(
    df_dm_enriched,
    genre_level_selector,
    genre_values,
    section_selector,
    sections_available,
    valid_sections,
):
    df_genre_filtered = df_dm_enriched
    sel = genre_level_selector.value
    if sel and sel in df_dm_enriched.columns and sel != "(none)" and genre_values.value:
        df_genre_filtered = df_genre_filtered.filter(
            pl.col(sel).is_in(genre_values.value)
        )

    allowed_sections = section_selector.value or sections_available
    allowed_effective = [
        s for s in allowed_sections if not valid_sections or s in valid_sections
    ]
    if allowed_effective:
        df_genre_filtered = df_genre_filtered.filter(
            pl.col("section").is_in(allowed_effective)
        )
    return df_genre_filtered, sel


@app.cell
def _(conn_agg, create_aggregated_df, df_genre_filtered, end_agg):
    agg_conn_df = create_aggregated_df(df_genre_filtered, conn_agg.value).filter(
        pl.col("タイプ") == "接続表現"
    )
    agg_end_df = create_aggregated_df(df_genre_filtered, end_agg.value).filter(
        pl.col("タイプ") == "文末表現"
    )
    aggregated_df = pl.concat([agg_conn_df, agg_end_df])
    return (aggregated_df,)


@app.cell
def _(
    add_selected_genre,
    aggregated_df,
    filter_types,
    genre_level_selector,
    include_conn_toggle,
    include_end_toggle,
):
    # This is the single DF that all analyses will use.
    # It respects aggregation level, type filters, and genre selection.
    _types_ok = filter_types(
        aggregated_df, include_conn_toggle.value, include_end_toggle.value
    )
    _preferred = (
        genre_level_selector.value if genre_level_selector.value != "(none)" else None
    )
    df_for_analysis = add_selected_genre(_types_ok, preferred=_preferred)
    df_for_analysis
    return (df_for_analysis,)


@app.cell
def _(df_for_analysis, group_cols_for, group_mode, pmi_sorted):
    gcols = group_cols_for(group_mode.value)
    pmi_df_viz = pmi_sorted(df_for_analysis, group_cols=gcols).sort("pmi", descending=True)
    mo.vstack(
        [
            mo.md("## PMI and transitions (with grouping)"),
            mo.ui.table(pmi_df_viz, selection=None, max_columns=None),
        ]
    )
    return (pmi_df_viz,)


@app.cell
def _():
    top_n = mo.ui.slider(
        label="Top-N expressions (overall)",
        start=10,
        stop=500,
        value=100,
        step=10,
        include_input=True,
        show_value=True,
    )

    mo.vstack(
        [
            mo.md("### Overall top expressions"),
            top_n,
        ]
    )
    return


@app.cell
def _(df_dm_enriched, genre_level_selector):
    col = genre_level_selector.value
    if not col or col not in df_dm_enriched.columns or col == "(none)":
        col = (
            "ジャンル"
            if "ジャンル" in df_dm_enriched.columns
            else next(
                (
                    c
                    for c in ["genre1", "genre2", "corpus"]
                    if c in df_dm_enriched.columns
                ),
                None,
            )
        )
    if col is None:
        mo.md("No genre column available to plot.")
    dm_genre_counts = (
        df_dm_enriched.filter(pl.col(col).is_not_null())
        .group_by(col)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    genre_topn_chart = (
        alt.Chart(dm_genre_counts)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="DM matches"),
            y=alt.Y(f"{col}:N", sort="-x", title="Genre"),
            tooltip=[alt.Tooltip("count:Q", title="DM matches")],
        )
        .properties(width=700, height=400)
    )
    mo.vstack(
        [
            mo.md("### Genre distribution (by DM matches)"),
            mo.ui.table(dm_genre_counts, selection=None, label="Counts by genre"),
            genre_topn_chart,
        ]
    )
    return


@app.cell
def _(pmi_df_viz):
    pmi_table = mo.ui.table(
        pmi_df_viz.sort("transition_conn_to_end", descending=True),
        selection='single',
        label="PMI and Transitions (select a row to see examples)"
    )

    mo.vstack(
        [
            mo.md("### PMI and transitions (with grouping)"),
            pmi_table,
        ]
    )
    return (pmi_table,)


@app.cell
def _():
    return


@app.cell(column=2)
def _():
    mo.md(
        r"""
    ## 対応分析

    > 図上の節と表現の相対的位置関係に意味がなく、その角度が近いものが類似していることになっている。
    """
    )
    return


@app.cell
def _(df_dm_enriched):
    genre_candidates = [
        c for c in ["genre1", "genre2", "corpus", "ジャンル"] if c in df_dm_enriched.columns
    ]
    genre_level_selector = mo.ui.dropdown(
        options=(["(none)"] + genre_candidates) if genre_candidates else ["(none)"],
        value=genre_candidates[1] if genre_candidates else "(none)",
        label="Genre granularity",
        searchable=True,
    )
    include_conn_toggle = mo.ui.switch(True, label="Include 接続表現")
    include_end_toggle = mo.ui.switch(True, label="Include 文末表現")
    group_mode = mo.ui.radio(
        options=["none", "section", "genre", "section+genre"],
        value="section",
        label="Group by (PMI/CA)",
    )
    ca_mode_selector = mo.ui.radio(
        options=["Expressions", "Collocations"],
        value="Expressions",
        label="CA Analysis Mode",
    )
    ca_min_freq = mo.ui.slider(
        label="CA: Min Freq per Item",
        start=1,
        stop=10000,
        value=1,
        step=1,
        show_value=True,
        include_input=True,
    )
    mo.vstack(
        [
            mo.hstack([genre_level_selector, include_conn_toggle, include_end_toggle, group_mode]),
            mo.hstack([ca_mode_selector, ca_min_freq]),
        ]
    )
    return (
        ca_min_freq,
        ca_mode_selector,
        genre_level_selector,
        group_mode,
        include_conn_toggle,
        include_end_toggle,
    )


@app.cell
def _(CE_LEVELS, SFE_LEVELS, df_dm_enriched):
    # Ensure parsed 機能_* columns exist
    df_expanded = expand_function_columns(df_dm_enriched)

    def get_available_options(df, dm_type: str, levels: list[str]) -> list[str]:
        """From a list of levels, return those that are available for the given dm_type."""
        type_df = df.filter(pl.col("タイプ") == dm_type)
        if type_df.is_empty():
            return []

        options: list[str] = []
        for lvl in levels:
            if lvl == "表現":
                # 表現 is always acceptable
                options.append(lvl)
                continue
            if lvl in df.columns:
                # include level only if there are non-null values for this DM type
                if type_df.filter(pl.col(lvl).is_not_null()).height > 0:
                    options.append(lvl)
        return options

    # Preferred ordering: 機能_* levels first (hierarchical), then 表現
    conn_options_ordered =  ["表現"] + [lvl for lvl in CE_LEVELS if lvl != "表現"]
    end_options_ordered = ["表現"] + [lvl for lvl in SFE_LEVELS if lvl != "表現"]

    conn_options = get_available_options(df_expanded, "接続表現", conn_options_ordered)
    end_options = get_available_options(df_expanded, "文末表現", end_options_ordered)

    conn_default = "表現"
    end_default = "表現"

    conn_agg = mo.ui.radio(
        options=conn_options,
        value=conn_default,
        label="接続表現：集計レベル (4 levels: cumulative hierarchy + expression)",
    )
    end_agg = mo.ui.radio(
        options=end_options,
        value=end_default,
        label="文末表現：集計レベル (3 levels: cumulative hierarchy + expression)",
    )

    mo.hstack([conn_agg, end_agg])
    return conn_agg, end_agg


@app.cell
def _(ca_min_freq, ca_mode_selector, df_for_analysis, group_mode):
    if df_for_analysis is None or (hasattr(df_for_analysis, "is_empty") and df_for_analysis.is_empty()):
        mo.md("No data available for Correspondence Analysis.")
        mo.stop(True)

    # 1. Prepare df_ca and expr_col_ca based on mode
    if ca_mode_selector.value == "Collocations":
        # Create collocation pairs
        conn_df = df_for_analysis.filter(pl.col("タイプ") == "接続表現").rename({"表現": "接続表現"})
        end_df = df_for_analysis.filter(pl.col("タイプ") == "文末表現").select(["sentence_id", "表現"]).rename({"表現": "文末表現"})

        # Join on sentence_id to get pairs.
        df_ca = conn_df.join(end_df, on="sentence_id", how="inner")

        # Create the collocation pair column
        df_ca = df_ca.with_columns(
            pl.concat_str([pl.col("接続表現"), pl.col("文末表現")], separator="→").alias("collocation")
        )
        expr_col_ca = "collocation"
    else:  # "Expressions" mode
        df_ca = df_for_analysis
        expr_col_ca = "表現"

    # No pre-filtering: compute CA on full data; visualization will apply min_freq.

    def get_ca_stats_title(df: pl.DataFrame, base_title: str, mode: str, expr_col: str) -> str:
        if df is None or df.is_empty():
            return f"{base_title} (No data)"

        n_sentences = int(df.select(pl.col("sentence_id")).n_unique())
        n_sections = int(df.select(pl.col("section")).n_unique())

        if mode == "Collocations":
            n_items = int(df.select(pl.col(expr_col)).n_unique())
            return f"{base_title} (sentences: {n_sentences}, sections: {n_sections}, collocation types: {n_items})"
        else:  # Expressions
            conn_df = df.filter(pl.col("タイプ") == "接続表現")
            n_conn = int(conn_df.select(pl.col(expr_col)).n_unique()) if not conn_df.is_empty() else 0

            end_df = df.filter(pl.col("タイプ") == "文末表現")
            n_end = int(end_df.select(pl.col(expr_col)).n_unique()) if not end_df.is_empty() else 0

            return f"{base_title} (sentences: {n_sentences}, sections: {n_sections}, connective types: {n_conn}, sentence-final types: {n_end})"

    def _cat_for_subset(mode: str) -> str | list[str]:
        # Within a fixed genre subset, use sections as categories to ensure multiple rows
        if mode in ("none", "section", "genre"):
            return "section"
        return ["section", "_selected_genre"]

    # Unique values of the currently selected genre column (added as _selected_genre)
    try:
        genre_vals = (
            df_ca.select(pl.col("_selected_genre"))
            .drop_nulls()
            .unique()
            .to_series()
            .to_list()
            if "_selected_genre" in df_ca.columns
            else []
        )
    except Exception:
        genre_vals = []

    views = [mo.md("### Correspondence Analysis (by selected genre)")]

    # Determine category_for_ca based on grouping mode
    if group_mode.value in ("none", "section"):
        category_for_ca = "section"
    elif group_mode.value == "genre":
        category_for_ca = "section"
    else:
        category_for_ca = ["section", "_selected_genre"]

    try:
        ca_plot_title = get_ca_stats_title(df_ca, "#### CA on all data", ca_mode_selector.value, expr_col_ca)
        views.append(mo.md(ca_plot_title))
        views.append(visualize_ca(df_ca, category_col=category_for_ca, expr_col=expr_col_ca, min_freq=ca_min_freq.value))
    except Exception as e:
        views.append(mo.md(f"CA unavailable for this selection: {e}"))

    if genre_vals:
        cat_col = _cat_for_subset(group_mode.value)
        for g in genre_vals:
            subset = df_ca.filter(pl.col("_selected_genre") == g)
            try:
                ca_plot_title = get_ca_stats_title(subset, f"#### Genre: {g}", ca_mode_selector.value, expr_col_ca)
                views.append(mo.md(ca_plot_title))
                views.append(visualize_ca(subset, category_col=cat_col, expr_col=expr_col_ca, min_freq=ca_min_freq.value))
            except Exception as e:
                views.append(mo.md(f"CA unavailable for genre '{g}': {e}"))
        mo.vstack(views)

    mo.vstack(views)
    return


@app.cell
def _(df_for_analysis, stat_category_selector):
    import prince
    from scipy.spatial import procrustes
    from scipy.spatial.distance import pdist
    from skbio.stats.distance import mantel
    from functools import reduce

    # Helper function to run CA and get coordinates
    def run_ca_and_get_coords(df_pl: pl.DataFrame, category_col: str, expr_col: str, n_components=2):
        if df_pl is None or df_pl.is_empty():
            return None

        ct_pl = build_contingency_table(df_pl, category_col=category_col, expr_col=expr_col)

        index_col = category_col if not isinstance(category_col, (list, tuple)) else " / ".join(category_col)
        if ct_pl.is_empty() or index_col not in ct_pl.columns:
            return None

        num_rows = ct_pl.height
        expr_cols = [c for c in ct_pl.columns if c != index_col]
        num_cols = len(expr_cols)
        if num_rows < 2 or num_cols < 2:
            return None

        max_comps = min(num_rows - 1, num_cols - 1)
        n_comp = min(2, max_comps)
        if n_comp <= 0:
            return None

        ct = ct_pl.to_pandas().set_index(index_col)
        ca = prince.CA(
            n_components=n_comp,
            n_iter=10,
            copy=True,
            check_input=True,
            engine="sklearn",
            random_state=42,
        ).fit(ct)
        return ca.row_coordinates(ct).sort_index()

    # Helper to run a single pairwise comparison
    def run_pairwise_comparison(coords1, coords2, name1, name2, common_sections):
        if coords1 is None or coords2 is None:
            return None, mo.md(f"**Cannot compare {name1} and {name2}**: one or both have no data.")

        # Align to common sections
        c1_aligned = coords1.loc[common_sections]
        c2_aligned = coords2.loc[common_sections]

        # Procrustes Analysis
        mtx1 = c1_aligned.to_numpy()
        mtx2 = c2_aligned.to_numpy()
        mtx1_p, mtx2_p, disparity = procrustes(mtx1, mtx2)

        # Mantel Test
        dist1 = pdist(mtx1, metric='euclidean')
        dist2 = pdist(mtx2, metric='euclidean')
        corr, p_value, n = mantel(dist1, dist2, method='pearson', permutations=999)

        # Prepare results
        md_out = mo.md(f'''
        #### {name1} vs. {name2}
        - **Procrustes Disparity (M²):** `{disparity:.4f}`
        - **Mantel Correlation (r):** `{corr:.4f}` (p-value: `{p_value:.4f}`)
        ''')

        procrustes_data = {
            "sections": common_sections,
            "matrix1_aligned": mtx1_p,
            "matrix2_aligned": mtx2_p,
        }
        return procrustes_data, md_out

    # --- Main logic ---
    comparison_outputs = [mo.md("### Quantitative Pairwise Comparison of CA Plots")]
    pairwise_procrustes_data = {}

    try:
        category_col = stat_category_selector.value

        # 1. Prepare all three dataframes
        df_conn = df_for_analysis.filter(pl.col("タイプ") == "接続表現")
        df_sfe = df_for_analysis.filter(pl.col("タイプ") == "文末表現")

        conn_coll_df = df_for_analysis.filter(pl.col("タイプ") == "接続表現").rename({"表現": "接続表現"})
        end_coll_df = df_for_analysis.filter(pl.col("タイプ") == "文末表現").select(["sentence_id", "表現"]).rename({"表現": "文末表現"})
        df_coll = conn_coll_df.join(end_coll_df, on="sentence_id", how="inner").with_columns(
            pl.concat_str([pl.col("接続表現"), pl.col("文末表現")], separator="→").alias("collocation")
        )

        # 2. Run CA on each
        coords_conn = run_ca_and_get_coords(df_conn, category_col, "表現")
        coords_sfe = run_ca_and_get_coords(df_sfe, category_col, "表現")
        coords_coll = run_ca_and_get_coords(df_coll, category_col, "collocation")

        # 3. Find common sections across all valid results
        valid_coords = [c for c in [coords_conn, coords_sfe, coords_coll] if c is not None]
        if len(valid_coords) < 2:
             comparison_outputs.append(mo.md("Fewer than two analyses produced valid CA results. Cannot compare."))
        else:
            common_sections = reduce(lambda a, b: a.intersection(b), [c.index for c in valid_coords])
            if len(common_sections) < 3:
                comparison_outputs.append(mo.md(f"Fewer than 3 common sections ({len(common_sections)}) across all analyses. Cannot run comparison."))
            else:
                # 4. Run pairwise comparisons
                data1, md1 = run_pairwise_comparison(coords_conn, coords_sfe, "Connectives", "Sentence-Finals", common_sections)
                if data1:
                    pairwise_procrustes_data['Connectives vs. Sentence-Finals'] = data1
                comparison_outputs.append(md1)

                data2, md2 = run_pairwise_comparison(coords_conn, coords_coll, "Connectives", "Collocations", common_sections)
                if data2:
                    pairwise_procrustes_data['Connectives vs. Collocations'] = data2
                comparison_outputs.append(md2)

                data3, md3 = run_pairwise_comparison(coords_sfe, coords_coll, "Sentence-Finals", "Collocations", common_sections)
                if data3:
                    pairwise_procrustes_data['Sentence-Finals vs. Collocations'] = data3
                comparison_outputs.append(md3)

    except Exception as e:
        comparison_outputs.append(mo.Html(f"<pre style='color: red;'>An error occurred: {e}</pre>"))

    mo.vstack(comparison_outputs)
    return (pairwise_procrustes_data,)


@app.cell
def _(pairwise_procrustes_data):
    import numpy as np

    def display_residuals(data, title):
        display_elements = [mo.md(f"#### Residuals for: {title}")]

        sections = data["sections"]
        m1 = data["matrix1_aligned"]
        m2 = data["matrix2_aligned"]
        residuals = np.sqrt(np.sum((m1 - m2)**2, axis=1))

        df_residuals = pl.DataFrame({
            "section": sections,
            "residual": residuals
        }).sort("residual", descending=True)

        display_elements.append(mo.ui.table(df_residuals, selection=None))

        chart = alt.Chart(df_residuals.to_pandas()).mark_bar().encode(
            x=alt.X('residual:Q', title='Residual (Mismatch)'),
            y=alt.Y('section:N', sort='-x', title='Section')
        ).properties(
            title=f"Procrustes Mismatch for {title}"
        )
        display_elements.append(chart)
        return display_elements

    residual_outputs = [mo.md("### Procrustes Residual Analysis: Mismatches by Section")]

    if not pairwise_procrustes_data:
        residual_outputs.append(mo.md("_(Comparison analysis did not run or did not produce data for residuals.)_"))
    else:
        for title, data in pairwise_procrustes_data.items():
            residual_outputs.extend(display_residuals(data, title))

    mo.vstack(residual_outputs)
    return


@app.cell
def _():
    mo.md("""## 統計テスト（カイ二乗）""")
    return


@app.cell
def _(df_for_analysis, group_mode):
    candidates = [
        c
        for c in [
            "section",
            "_selected_genre",
            "ジャンル",
            "genre1",
            "genre2",
            "genre3",
        ]
        if c in df_for_analysis.columns
    ]
    if group_mode.value == "genre" and "_selected_genre" in candidates:
        default_cat = "_selected_genre"
    else:
        default_cat = (
            "section"
            if "section" in candidates
            else (candidates[0] if candidates else "section")
        )
    stat_category_selector = mo.ui.dropdown(
        options=candidates or ["section"],
        value=default_cat,
        label="Category axis (Chi2/Keyness)",
        searchable=True,
    )
    chi_top_k = mo.ui.slider(
        label="Chi-square heatmap: top-N expressions",
        start=10,
        stop=200,
        value=60,
        step=10,
        include_input=True,
        show_value=True,
    )
    keyness_min_count = mo.ui.slider(
        label="Keyness: minimum total count per expression",
        start=1,
        stop=200,
        value=5,
        step=1,
        include_input=True,
        show_value=True,
    )
    keyness_top_n = mo.ui.slider(
        label="Keyness: Top-N (by LLR)",
        start=10,
        stop=200,
        value=50,
        step=10,
        include_input=True,
        show_value=True,
    )
    keyness_correction = mo.ui.dropdown(
        options=["fdr_bh", "bonferroni", None],
        value="fdr_bh",
        label="Keyness: correction method",
    )
    mo.hstack([stat_category_selector, chi_top_k, keyness_min_count, keyness_top_n, keyness_correction])
    return (
        chi_top_k,
        keyness_correction,
        keyness_min_count,
        keyness_top_n,
        stat_category_selector,
    )


@app.cell
def _(chi_top_k, df_for_analysis, stat_category_selector):
    # Chi-squared standardized residuals and heatmap (drop all-zero rows/cols for stability)
    residuals, chi2_stat, dof, p_value, cramers_v = chi2_standardized_residuals(
        df_for_analysis,
        category_col=stat_category_selector.value,
        expr_col="表現",
        drop_zeros=True,
    )
    if residuals.is_empty():
        _view = mo.md("No data for Chi-squared residuals with current selection.")
    else:
        scores = (
            residuals.group_by("表現")
            .agg(pl.col("std_resid").abs().sum().alias("_score"))
            .sort("_score", descending=True)
            .head(chi_top_k.value)
        )
        keep_exprs = set(scores["表現"].to_list())
        resid_top = residuals.filter(pl.col("表現").is_in(list(keep_exprs)))

        heat = (
            alt.Chart(resid_top)
            .mark_rect()
            .encode(
                x=alt.X("表現:N", sort=None, title="Expression"),
                y=alt.Y(
                    f"{stat_category_selector.value}:N", sort="-x", title="Category"
                ),
                color=alt.Color(
                    "std_resid:Q",
                    title="Std resid",
                    scale=alt.Scale(scheme="blueorange", domainMid=0),
                ),
                tooltip=[
                    alt.Tooltip(f"{stat_category_selector.value}:N", title="Category"),
                    alt.Tooltip("表現:N", title="Expression"),
                    alt.Tooltip("observed:Q", title="Obs"),
                    alt.Tooltip("expected:Q", title="Exp", format=".2f"),
                    alt.Tooltip("std_resid:Q", title="Std resid", format=".2f"),
                ],
            )
            .properties(width=900, height=500)
        )
        _view = mo.vstack(
            [
                mo.md(
                    f"### カイ二乗検定: χ²={chi2_stat:.2f}, df={dof}, p={p_value:.3g}, V={cramers_v:.3f}",
                ),
                heat,
            ]
        )
    _view
    return


@app.cell(column=3)
def _(CE_LEVELS, SFE_LEVELS, df_dm_enriched):
    df_fx = expand_function_columns(df_dm_enriched)

    def _values_for_level(dfa: pl.DataFrame, typ: str, level: str) -> list[str]:
        if level not in dfa.columns:
            return []
        return (
            dfa.filter(pl.col("タイプ") == typ)
            .select(pl.col(level))
            .drop_nulls()
            .unique()
            .to_series()
            .to_list()
        )

    conn_levels = [c for c in CE_LEVELS if c in df_fx.columns]
    end_levels = [c for c in SFE_LEVELS if c in df_fx.columns]

    values_per_conn_level = {lvl: _values_for_level(df_fx, "接続表現", lvl) for lvl in conn_levels}
    values_per_end_level = {lvl: _values_for_level(df_fx, "文末表現", lvl) for lvl in end_levels}

    q = mo.ui.text(label="Search span_text (regex)", value="")
    k = mo.ui.slider(
        label="Max rows", start=10, stop=2000, value=200, step=10, include_input=True, show_value=True
    )
    show_para_toggle = mo.ui.switch(label="Show full paragraph")

    mo.vstack(
        [
            mo.md("# 共起検索\n正規表現あるいは個別表現・共起表現で検索できます。"),
            mo.md("## 正規表現の指定"),
            mo.hstack([q, k, show_para_toggle]),
        ]
    )
    return (
        df_fx,
        k,
        q,
        show_para_toggle,
        values_per_conn_level,
        values_per_end_level,
    )


@app.cell
def _(conn_agg, end_agg, values_per_conn_level, values_per_end_level):
    conn_value_options = ["(any)", "(None)"] + sorted(values_per_conn_level.get(conn_agg.value, []))
    end_value_options = ["(any)", "(None)"] + sorted(values_per_end_level.get(end_agg.value, []))
    conn_value = mo.ui.dropdown(options=conn_value_options, value="(any)", label="接続表現：", searchable=True)
    end_value = mo.ui.dropdown(options=end_value_options, value="(any)", label="文末表現：", searchable=True)
    mo.hstack([mo.md("## 接続・文末表現の指定"), conn_value, end_value])
    return conn_value, end_value


@app.cell
def _(
    conn_agg,
    conn_value,
    df_all_sentences,
    df_genre_filtered,
    end_agg,
    end_value,
    genre_values,
    k,
    pmi_table,
    q,
    sel,
    show_para_toggle,
):
    c_col, c_val = (conn_agg.value, conn_value.value)
    e_col, e_val = (end_agg.value, end_value.value)
    query_str = q.value

    # If a row is selected in the PMI table, override the values
    # pmi_table.value is a dataframe of the selected rows
    if not pmi_table.value.is_empty():
        try:
            # Get the first (and only) selected row as a dictionary
            selected_row_dict = pmi_table.value.to_dicts()[0]
            print(selected_row_dict)
            c_col, c_val = "表現", selected_row_dict.get("接続表現")
            e_col, e_val = "表現", selected_row_dict.get("文末表現")
            query_str = "" # Clear regex search for clarity
        except IndexError:
            print("IndexError")
            # Should not happen if not is_empty(), but as a safeguard
            pass

    # Final check to not pass "(any)" to the search function
    final_c_val = c_val if c_val != "(any)" else None
    final_e_val = e_val if e_val != "(any)" else None

    # In regex mode (only query is provided), don't require collocations.
    is_regex_mode = bool(query_str) and not final_c_val and not final_e_val

    print(final_c_val, final_e_val, query_str)

    df_all_sentences_filtered = df_all_sentences
    if sel and sel in df_all_sentences.columns and sel != "(none)" and genre_values.value:
        df_all_sentences_filtered = df_all_sentences_filtered.filter(
            pl.col(sel).is_in(genre_values.value)
        )

    html = collocation_sentences_html(
        df_genre_filtered,
        query=query_str,
        case_sensitive=False,
        max_rows=k.value,
        conn_col=c_col if final_c_val else None,
        conn_value=final_c_val,
        end_col=e_col if final_e_val else None,
        end_value=final_e_val,
        require_both=not is_regex_mode,
        show_paragraph=show_para_toggle.value,
        all_sentences_df=df_all_sentences_filtered,
    )
    mo.Html(html)
    return


@app.cell
def _(sampled_sections_df):
    import xxhash

    # Construct a dataframe of all sentences to allow searching for sentences *without* DMs.
    entries = []
    for row in sampled_sections_df.iter_rows(named=True):
        section_text = row.get("text") or ""
        meta_base = {k: v for k, v in row.items() if k != "text"}
        for para_text in section_text.split("\n\n"):
            para_text = para_text.strip()
            if not para_text:
                continue
            meta = meta_base.copy()
            meta["paragraph_text"] = para_text
            for line in para_text.splitlines():
                if not line.strip():
                    continue
                bn = str(meta.get("basename") or "")
                sec = str(meta.get("section") or "")
                sid_key = f"{bn}|{sec}|{line}"
                sid = (
                    xxhash.xxh64(sid_key.encode("utf-8")).intdigest()
                    & 0x7FFFFFFFFFFFFFFF
                )
                entry = meta.copy()
                entry["sentence_id"] = sid
                entry["sentence_text"] = line
                entries.append(entry)
    df_all_sentences = pl.DataFrame(entries)
    return (df_all_sentences,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
