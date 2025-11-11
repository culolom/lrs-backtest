# === 綜合績效報表（含CSS美化） ===

st.markdown("""
<style>
.custom-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1.2em;
    font-family: "Noto Sans TC", "Microsoft JhengHei", sans-serif;
    box-shadow: 0 3px 8px rgba(0,0,0,0.05);
    border-radius: 10px;
    overflow: hidden;
}
.custom-table th {
    background-color: #f5f6fa;
    color: #2c3e50;
    text-align: center;
    padding: 12px;
    font-weight: 700;
    border-bottom: 2px solid #e0e0e0;
}
.custom-table td {
    text-align: center;
    padding: 10px;
    border-bottom: 1px solid #e9e9e9;
    font-size: 15px;
}
.custom-table tr:nth-child(even) td {
    background-color: #fafbfc;
}
.custom-table tr:hover td {
    background-color: #f1f9ff;
}
.custom-table .section-title td {
    background-color: #eef4ff;
    color: #1a237e;
    font-weight: 700;
    font-size: 16px;
    text-align: left;
    padding: 10px 15px;
    border-top: 2px solid #cfd8dc;
}
</style>
""", unsafe_allow_html=True)

html_table = f"""
<table class='custom-table'>
<thead><tr><th>指標名稱</th><th>LRS 策略</th><th>Buy & Hold</th></tr></thead>
<tbody>
<tr><td>最終資產</td><td>{equity_lrs_final:,.0f} 元</td><td>{equity_bh_final:,.0f} 元</td></tr>
<tr><td>總報酬</td><td>{final_return_lrs:.2%}</td><td>{final_return_bh:.2%}</td></tr>
<tr><td>年化報酬</td><td>{cagr_lrs:.2%}</td><td>{cagr_bh:.2%}</td></tr>
<tr><td>最大回撤</td><td>{mdd_lrs:.2%}</td><td>{mdd_bh:.2%}</td></tr>
<tr><td>年化波動率</td><td>{vol_lrs:.2%}</td><td>{vol_bh:.2%}</td></tr>
<tr><td>夏普值</td><td>{sharpe_lrs:.2f}</td><td>{sharpe_bh:.2f}</td></tr>
<tr><td>索提諾值</td><td>{sortino_lrs:.2f}</td><td>{sortino_bh:.2f}</td></tr>
<tr class='section-title'><td colspan='3'>💹 交易統計</td></tr>
<tr><td>買進次數</td><td>{buy_count}</td><td>—</td></tr>
<tr><td>賣出次數</td><td>{sell_count}</td><td>—</td></tr>
</tbody></table>
"""

st.markdown(html_table, unsafe_allow_html=True)
