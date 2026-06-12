import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MAPL+ | Portfolio Pricing Intelligence",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    import os
    base = os.path.dirname(os.path.abspath(__file__))

    sku_overview    = pd.read_csv(os.path.join(base, 'sku_overview.csv'))
    price_lookup_df = pd.read_csv(os.path.join(base, 'price_lookup.csv'))
    matrix_df       = pd.read_csv(os.path.join(base, 'cannibalization_matrix.csv'), index_col=0)
    elast_df        = pd.read_csv(os.path.join(base, 'elasticity_per_sku.csv'))
    forecast_df     = pd.read_csv(os.path.join(base, 'xgb_forecast.csv'))
    ...
    sku_meta = (sku_overview[['SKU_ID','SKU','Brand','SKU_Category']]
                .drop_duplicates().reset_index(drop=True))
    
    elast_lookup = dict(zip(elast_df['SKU_ID'], elast_df['FinalElasticity']))
    price_lookup = dict(zip(price_lookup_df['SKU_ID'], price_lookup_df['NormalPrice']))

    return sku_overview, sku_meta, matrix_df, elast_lookup, price_lookup, forecast_df

df, sku_meta, matrix_df, elast_lookup, price_lookup, forecast_df = load_data()

# ── Simulator engine ─────────────────────────────────────────────────────────
def simulate_portfolio_impact(sku_a_id, discount_pct, branch, weeks=4):
    skus      = sku_meta['SKU_ID'].tolist()
    sku_name  = dict(zip(sku_meta['SKU_ID'], sku_meta['SKU']))
    sku_cat   = dict(zip(sku_meta['SKU_ID'], sku_meta['SKU_Category']))
    sku_brand = dict(zip(sku_meta['SKU_ID'], sku_meta['Brand']))

    elast_a      = elast_lookup.get(sku_a_id, -1.0)
    uplift_pct_a = elast_a * (-discount_pct)

    rows = []
    for week in range(1, weeks + 1):

        # ── Base forecast per minggu (XGBoost output) ──
        def get_base(sku_id):
            val = forecast_df[
                (forecast_df['SKU_ID']==sku_id) &
                (forecast_df['Branch']==branch) &
                (forecast_df['ForecastWeek']==week)
            ]['ForecastQty']
            return val.values[0] if len(val) > 0 else 0

        base_qty_a = get_base(sku_a_id)
        uplift_qty_a = base_qty_a * uplift_pct_a  # naik tiap minggu kalau trend naik

        for sku_b_id in skus:
            base_qty_b     = get_base(sku_b_id)
            normal_price_b = price_lookup.get(sku_b_id, 0)

            if sku_b_id == sku_a_id:
                adj_qty      = base_qty_b * (1 + uplift_pct_a)
                delta_qty    = adj_qty - base_qty_b
                disc_price   = normal_price_b * (1 - discount_pct)
                base_rev     = base_qty_b * normal_price_b
                adj_rev      = adj_qty * disc_price
                effect_type  = 'Promo Uplift'
                coef         = 0.0
            else:
                coef = float(matrix_df.loc[sku_a_id, sku_b_id]) if (
                    sku_a_id in matrix_df.index and
                    sku_b_id in matrix_df.columns) else 0.0

                loss_qty_b   = uplift_qty_a * coef
                adj_qty      = max(base_qty_b - loss_qty_b, 0)
                delta_qty    = adj_qty - base_qty_b
                base_rev     = base_qty_b * normal_price_b
                adj_rev      = adj_qty * normal_price_b
                effect_type  = 'Cannibalization' if coef > 0 else 'No Effect'

            rows.append({
                'Week':            week,
                'SKU_ID':          sku_b_id,
                'SKU':             sku_name.get(sku_b_id,''),
                'Brand':           sku_brand.get(sku_b_id,''),
                'Category':        sku_cat.get(sku_b_id,''),
                'BaseQty':         round(base_qty_b, 1),
                'AdjustedQty':     round(adj_qty, 1),
                'DeltaQty':        round(delta_qty, 1),
                'BaseRevenue':     round(base_rev, 0),
                'AdjustedRevenue': round(adj_rev, 0),
                'DeltaRevenue':    round(adj_rev - base_rev, 0),
                'EffectType':      effect_type,
                'CannibCoef':      coef,
            })

    weekly_df = pd.DataFrame(rows)
    summary   = (weekly_df.groupby(['SKU_ID','SKU','Brand','Category','EffectType'])
                          .agg(
                              TotalBaseQty  = ('BaseQty','sum'),
                              TotalAdjQty   = ('AdjustedQty','sum'),
                              TotalDeltaQty = ('DeltaQty','sum'),
                              TotalBaseRev  = ('BaseRevenue','sum'),
                              TotalAdjRev   = ('AdjustedRevenue','sum'),
                              TotalDeltaRev = ('DeltaRevenue','sum'),
                          ).reset_index())

    port_base      = summary['TotalBaseRev'].sum()
    port_adj       = summary['TotalAdjRev'].sum()
    port_delta     = port_adj - port_base
    port_delta_pct = port_delta / port_base * 100 if port_base > 0 else 0

    return weekly_df, summary, port_base, port_adj, port_delta, port_delta_pct

# ════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
    <h1 style='color:#E53935; margin-bottom:0'>MAPL+</h1>
    <p style='color:#666; font-size:16px; margin-top:4px'>
    Portfolio-Aware Pricing Intelligence Platform
    </p>
    <hr style='border:1px solid #eee'>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 What-If Simulator", "🔥 Cannibalization Heatmap", "📋 SKU Overview"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIMULATOR
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("What-If Pricing Simulator")
    st.caption("Simulasikan dampak promo satu SKU terhadap seluruh portfolio selama 4 minggu ke depan.")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        sku_options = dict(zip(sku_meta['SKU'], sku_meta['SKU_ID']))
        selected_sku_name = st.selectbox("Pilih SKU (Aggressor)", options=list(sku_options.keys()))
        selected_sku_id   = sku_options[selected_sku_name]

    with col2:
        selected_branch = st.selectbox("Branch", ['Jakarta','Surabaya','Bandung','Semarang'])

    with col3:
        discount_pct = st.slider("Discount (%)", min_value=5, max_value=50,
                                  value=20, step=5) / 100

    run_btn = st.button("🚀 Jalankan Simulasi", type="primary", use_container_width=True)

    if run_btn:
        weekly_df, summary, port_base, port_adj, port_delta, port_delta_pct = \
            simulate_portfolio_impact(selected_sku_id, discount_pct, selected_branch)

        # ── KPI Cards ────────────────────────────────────────────────────────
        st.markdown("---")
        k1, k2, k3, k4 = st.columns(4)

        sku_row = summary[summary['SKU_ID'] == selected_sku_id].iloc[0]
        cannib_rows = summary[summary['EffectType'] == 'Cannibalization']

        k1.metric("Portfolio ΔRevenue (4 minggu)",
                  f"Rp {port_delta:,.0f}",
                  f"{port_delta_pct:.1f}%",
                  delta_color="inverse")

        k2.metric("Uplift Qty SKU Target",
                  f"+{sku_row['TotalDeltaQty']:,.0f} unit",
                  f"dari {sku_row['TotalBaseQty']:,.0f} base")

        k3.metric("SKU Terdampak Cannibalization",
                  f"{len(cannib_rows)} SKU",
                  f"Total loss: {cannib_rows['TotalDeltaQty'].sum():,.0f} unit",
                  delta_color="inverse")

        k4.metric("Net Qty Change Portfolio",
                  f"{summary['TotalDeltaQty'].sum():,.0f} unit",
                  delta_color="normal")

        st.markdown("---")
        col_left, col_right = st.columns(2)

        # ── Chart: Revenue waterfall ─────────────────────────────────────────
        with col_left:
            affected = summary[summary['EffectType'] != 'No Effect'].copy()
            affected = affected.sort_values('TotalDeltaRev')

            colors_bar = ['#2196F3' if e == 'Promo Uplift' else '#E53935'
                          for e in affected['EffectType']]

            fig_bar = go.Figure(go.Bar(
                x=affected['TotalDeltaRev'],
                y=affected['SKU'].str[:28],
                orientation='h',
                marker_color=colors_bar,
                text=[f"Rp {v:,.0f}" for v in affected['TotalDeltaRev']],
                textposition='outside',
                textfont_size=9,
            ))
            fig_bar.update_layout(
                title='Revenue Impact per SKU (4 minggu)',
                xaxis_title='Delta Revenue (Rp)',
                height=420,
                margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor='white',
                xaxis=dict(gridcolor='#eee'),
            )
            fig_bar.add_vline(x=0, line_color='black', line_width=1)
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Chart: Weekly trend ──────────────────────────────────────────────
        with col_right:
            top_cannib = (cannib_rows.nsmallest(5, 'TotalDeltaRev')['SKU_ID'].tolist()
                          + [selected_sku_id])
            weekly_top = weekly_df[weekly_df['SKU_ID'].isin(top_cannib)]

            fig_line = go.Figure()
            colors_line = px.colors.qualitative.Set2
            for i, sku_id in enumerate(top_cannib):
                sub = weekly_top[weekly_top['SKU_ID'] == sku_id]
                sku_label = sku_meta[sku_meta['SKU_ID']==sku_id]['SKU'].values[0][:25]
                dash = 'solid' if sku_id == selected_sku_id else 'dot'
                fig_line.add_trace(go.Scatter(
                    x=sub['Week'], y=sub['AdjustedQty'],
                    mode='lines+markers',
                    name=sku_label,
                    line=dict(dash=dash, color=colors_line[i % len(colors_line)]),
                ))
            fig_line.update_layout(
                title='Projected Weekly Qty (Top 5 Affected + Target)',
                xaxis_title='Minggu ke-',
                yaxis_title='Projected Qty',
                height=420,
                margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor='white',
                xaxis=dict(gridcolor='#eee', tickvals=[1,2,3,4]),
                yaxis=dict(gridcolor='#eee'),
                legend=dict(font_size=9),
            )
            st.plotly_chart(fig_line, use_container_width=True)

        # ── Detail table ─────────────────────────────────────────────────────
        st.markdown("#### Detail per SKU")
        display = summary[['SKU','Brand','Category','EffectType',
                            'TotalBaseQty','TotalDeltaQty',
                            'TotalBaseRev','TotalDeltaRev']].copy()
        display.columns = ['SKU','Brand','Category','Effect',
                            'Base Qty','Δ Qty','Base Rev (Rp)','Δ Rev (Rp)']
        display = display.sort_values('Δ Rev (Rp)')

        def color_delta(val):
            if val < 0: return 'color: #E53935'
            elif val > 0: return 'color: #2E7D32'
            return ''

        st.dataframe(
            display.style.applymap(color_delta, subset=['Δ Qty','Δ Rev (Rp)'])
                         .format({'Base Qty': '{:,.0f}', 'Δ Qty': '{:+,.0f}',
                                  'Base Rev (Rp)': '{:,.0f}', 'Δ Rev (Rp)': '{:+,.0f}'}),
            use_container_width=True, height=400
        )

        # ── AI Recommendation ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🤖 AI Recommendation")

        net_ok = port_delta_pct > -2
        cannib_count = len(cannib_rows[cannib_rows['TotalDeltaRev'] < -500000])
        worst_victim = summary[summary['EffectType']=='Cannibalization'].nsmallest(1,'TotalDeltaRev')

        if port_delta_pct > 0:
            verdict = "✅ **Promo ini menguntungkan portfolio secara keseluruhan.**"
            rec = "Lanjutkan promo dengan monitoring mingguan."
        elif port_delta_pct > -3:
            verdict = "⚠️ **Promo ini memberikan dampak negatif ringan ke portfolio.**"
            rec = f"Pertimbangkan menurunkan discount ke {int(discount_pct*100)-5}% untuk mengurangi cannibalization."
        else:
            verdict = "🔴 **Promo ini berisiko tinggi — cannibalization melebihi uplift revenue.**"
            rec = "Tidak disarankan tanpa adjustment. Evaluasi ulang discount level atau batasi promo ke branch tertentu."

        wv_name = worst_victim['SKU'].values[0][:30] if len(worst_victim) > 0 else '-'
        wv_rev  = worst_victim['TotalDeltaRev'].values[0] if len(worst_victim) > 0 else 0

        st.info(f"""
{verdict}

**Portfolio Impact:** Rp {port_delta:,.0f} ({port_delta_pct:.1f}%) selama 4 minggu  
**SKU paling terdampak:** {wv_name} (Rp {wv_rev:,.0f})  
**Rekomendasi:** {rec}
        """)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — CANNIBALIZATION HEATMAP
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Cannibalization Heatmap")
    st.caption("Seberapa besar satu SKU berpotensi mengkanibal SKU lain saat mendapat promo.")

    filter_brand = st.multiselect("Filter Brand",
                                   options=sku_meta['Brand'].unique(),
                                   default=sku_meta['Brand'].unique().tolist())

    filtered_skus = sku_meta[sku_meta['Brand'].isin(filter_brand)]['SKU_ID'].tolist()
    sku_labels    = sku_meta[sku_meta['Brand'].isin(filter_brand)]['SKU'].str[:25].tolist()

    mat_filtered  = matrix_df.loc[filtered_skus, filtered_skus].copy()
    mat_filtered.index   = sku_labels
    mat_filtered.columns = sku_labels

    fig_heat = go.Figure(go.Heatmap(
        z=mat_filtered.values,
        x=mat_filtered.columns.tolist(),
        y=mat_filtered.index.tolist(),
        colorscale='YlOrRd',
        zmin=0, zmax=0.4,
        text=np.round(mat_filtered.values, 2),
        texttemplate='%{text}',
        textfont_size=9,
        colorbar=dict(title='Cannib. Coef'),
    ))
    fig_heat.update_layout(
        title='Cannibalization Coefficient Matrix<br><sup>Row = Aggressor (SKU diberi promo), Col = Victim</sup>',
        height=600,
        xaxis=dict(tickfont_size=9),
        yaxis=dict(tickfont_size=9, autorange='reversed'),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("**Interpretasi:** Nilai 0.30 artinya setiap 100 unit uplift SKU aggressor, SKU victim kehilangan ~30 unit demand.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — SKU OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("SKU Overview")

    branch_filter = st.selectbox("Branch", ['All','Jakarta','Surabaya','Bandung','Semarang'])
    df_view = df if branch_filter == 'All' else df[df['Branch'] == branch_filter]

    sku_summary = (df_view.groupby(['SKU_ID','SKU','Brand','SKU_Category'])
                          .agg(
                              TotalQty=('Qty','sum'),
                              TotalRevenue=('Total_Price','sum'),
                              TotalTx=('TransactionID','count'),
                              AvgDiscount=('DiscountPercentage','mean'),
                          ).reset_index()
                          .sort_values('TotalRevenue', ascending=False))

    sku_summary['AvgDiscount'] = (sku_summary['AvgDiscount'] * 100).round(1)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_rev = px.bar(sku_summary, x='TotalRevenue', y='SKU',
                          color='Brand', orientation='h',
                          color_discrete_map={'Richeese':'#E53935',
                                              'Richoco':'#8D6E63',
                                              'Nextar':'#1E88E5'},
                          title='Total Revenue per SKU')
        fig_rev.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_rev, use_container_width=True)

    with col_b:
        fig_qty = px.bar(sku_summary, x='TotalQty', y='SKU',
                          color='Brand', orientation='h',
                          color_discrete_map={'Richeese':'#E53935',
                                              'Richoco':'#8D6E63',
                                              'Nextar':'#1E88E5'},
                          title='Total Qty per SKU')
        fig_qty.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_qty, use_container_width=True)

    st.dataframe(
        sku_summary[['SKU','Brand','SKU_Category','TotalQty',
                     'TotalRevenue','TotalTx','AvgDiscount']]
                  .rename(columns={'SKU_Category':'Category',
                                   'TotalQty':'Total Qty',
                                   'TotalRevenue':'Revenue (Rp)',
                                   'TotalTx':'Transactions',
                                   'AvgDiscount':'Avg Disc %'})
                  .style.format({'Total Qty': '{:,.0f}',
                                 'Revenue (Rp)': '{:,.0f}',
                                 'Transactions': '{:,.0f}',
                                 'Avg Disc %': '{:.1f}%'}),
        use_container_width=True
    )
