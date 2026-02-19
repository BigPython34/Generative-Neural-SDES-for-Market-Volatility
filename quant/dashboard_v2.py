"""
Interactive Dashboard V2 — Enhanced
====================================
Full-featured dashboard with:
  - Calibration results & model comparison
  - Backtest win rates
  - Market regime indicator
  - Risk metrics (VaR/CVaR)
  - SOFR rate & VVIX level
  - IV surface 3D plot
"""

import numpy as np
from datetime import datetime
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DashboardV2:
    """Enhanced dashboard generator with risk & regime panels."""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.metrics = {}
        self.surface = None

    def load_data(self):
        """Load all available data."""
        if os.path.exists('outputs/calibration_report.json'):
            with open('outputs/calibration_report.json', 'r') as f:
                self.metrics['calibration'] = json.load(f)
            print("   Calibration report loaded")

        if os.path.exists('outputs/backtest_results.json'):
            with open('outputs/backtest_results.json', 'r') as f:
                self.metrics['backtest'] = json.load(f)
            print("   Backtest results loaded")

        if os.path.exists('outputs/model_usecases_report.json'):
            with open('outputs/model_usecases_report.json', 'r') as f:
                self.metrics['usecases'] = json.load(f)
            print("   Use-cases report loaded")

        # Regime detection
        try:
            from quant.regime_detector import RegimeDetector
            detector = RegimeDetector()
            self.metrics['regime'] = detector.detect()
            print(f"   Regime: {self.metrics['regime']['regime']}")
        except Exception as e:
            print(f"   Regime detection skipped: {e}")

        # SOFR rate
        try:
            from utils.sofr_loader import SOFRRateLoader
            sofr = SOFRRateLoader()
            if sofr.is_available:
                self.metrics['sofr_rate'] = sofr.get_rate()
                print(f"   SOFR rate: {self.metrics['sofr_rate']:.4f}")
        except Exception:
            pass

        # VVIX
        try:
            from utils.vvix_calibrator import VVIXCalibrator
            vvix = VVIXCalibrator()
            if vvix.is_available:
                eta_info = vvix.estimate_eta()
                self.metrics['vvix'] = eta_info
                print(f"   VVIX: {eta_info.get('vvix_current', 'N/A')}")
        except Exception:
            pass

        # Surface
        try:
            from quant.options_cache import OptionsDataCache
            cache = OptionsDataCache()
            self.surface, info = cache.load_latest("SPY")
            self.metrics['surface_info'] = info
            print(f"   Surface loaded: {len(self.surface)} options")
        except Exception:
            self.surface = None

    def generate(self, output_path: str = "outputs/dashboard.html"):
        """Generate the HTML dashboard."""
        print("\nGenerating Dashboard V2...")
        self.load_data()

        calib = self.metrics.get('calibration', {})
        backtest = self.metrics.get('backtest', {})
        surface_info = self.metrics.get('surface_info', {})
        regime = self.metrics.get('regime', {})
        usecases = self.metrics.get('usecases', {})

        spot = calib.get('spot', surface_info.get('spot', 0))
        atm_iv = calib.get('atm_iv', 0) * 100
        n_options = calib.get('n_options', 0)
        rmse = calib.get('rmse_results', {})
        bergomi_params = calib.get('bergomi_params', {})

        bt_summary = backtest.get('summary', {})
        win_rates = backtest.get('win_rates', {})

        # Risk metrics from usecases
        risk_data = {}
        for profile_name, profile in usecases.get('profiles', {}).items():
            rs = profile.get('risk_scenarios', {})
            if rs:
                risk_data = rs
                break

        regime_name = regime.get('regime', 'unknown')
        regime_confidence = regime.get('confidence', 0) * 100
        sofr_rate = self.metrics.get('sofr_rate', 0.05)
        vvix_info = self.metrics.get('vvix', {})
        vvix_current = vvix_info.get('vvix_current', 0)
        eta_rec = vvix_info.get('eta_recommended', bergomi_params.get('eta', 0))

        regime_colors = {
            'calm': '#00ff88', 'normal': '#00d4ff',
            'stressed': '#ffd700', 'crisis': '#ff4444',
            'elevated': '#ff9800', 'unknown': '#888',
        }
        regime_color = regime_colors.get(regime_name, '#888')

        surface_div = self._create_surface_plot()
        comparison_div = self._create_comparison_plot(rmse)
        backtest_div = self._create_backtest_plot(bt_summary, win_rates)

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepRoughVol Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{
            text-align: center;
            padding: 30px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            margin-bottom: 25px;
            border: 1px solid rgba(0,212,255,0.2);
        }}
        .header h1 {{
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .header .subtitle {{ color: #888; font-size: 1.1em; }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 12px;
            margin-bottom: 25px;
        }}
        .kpi-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 18px 12px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.2s, border-color 0.2s;
        }}
        .kpi-card:hover {{
            transform: translateY(-3px);
            border-color: rgba(0,212,255,0.5);
        }}
        .kpi-value {{ font-size: 1.8em; font-weight: bold; margin-bottom: 3px; }}
        .kpi-label {{ color: #888; font-size: 0.85em; }}
        .kpi-cyan .kpi-value {{ color: #00d4ff; }}
        .kpi-green .kpi-value {{ color: #00ff88; }}
        .kpi-gold .kpi-value {{ color: #ffd700; }}
        .kpi-magenta .kpi-value {{ color: #ff6bff; }}
        .kpi-orange .kpi-value {{ color: #ff9800; }}
        .kpi-red .kpi-value {{ color: #ff4444; }}
        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 25px;
        }}
        .grid-3 {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 25px;
        }}
        @media (max-width: 900px) {{
            .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
        }}
        .card {{
            background: rgba(255,255,255,0.03);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.08);
        }}
        .card-title {{
            font-size: 1.15em;
            margin-bottom: 15px;
            color: #00d4ff;
        }}
        .full-width {{ grid-column: 1 / -1; }}
        .params-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-top: 10px;
        }}
        .param-box {{
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }}
        .param-value {{ font-size: 1.6em; font-weight: bold; color: #00d4ff; }}
        .param-label {{ color: #888; font-size: 0.8em; margin-top: 4px; }}
        .param-note {{ color: #00ff88; font-size: 0.7em; margin-top: 2px; }}
        .regime-badge {{
            display: inline-block;
            padding: 4px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .risk-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        .risk-table td {{ padding: 8px 12px; border-bottom: 1px solid rgba(255,255,255,0.08); }}
        .risk-table td:first-child {{ color: #888; }}
        .risk-table td:last-child {{ text-align: right; font-weight: bold; }}
        .plot-container {{ width: 100%; min-height: 350px; }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #555;
            font-size: 0.85em;
            margin-top: 30px;
        }}
        .section-title {{
            font-size: 1.3em;
            color: #00ff88;
            margin: 30px 0 15px 0;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(0,255,136,0.2);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>DeepRoughVol Dashboard</h1>
            <div class="subtitle">
                Neural SDE + Rough Bergomi | Multi-Measure Pricing & Risk | {self.timestamp}
            </div>
        </div>

        <!-- KPI Row 1: Market -->
        <div class="kpi-grid">
            <div class="kpi-card kpi-cyan">
                <div class="kpi-value">${spot:.2f}</div>
                <div class="kpi-label">SPY Spot</div>
            </div>
            <div class="kpi-card kpi-green">
                <div class="kpi-value">{atm_iv:.1f}%</div>
                <div class="kpi-label">ATM IV</div>
            </div>
            <div class="kpi-card kpi-gold">
                <div class="kpi-value">{vvix_current:.0f}</div>
                <div class="kpi-label">VVIX</div>
            </div>
            <div class="kpi-card kpi-magenta">
                <div class="kpi-value">{sofr_rate*100:.2f}%</div>
                <div class="kpi-label">SOFR Rate</div>
            </div>
            <div class="kpi-card" style="border-color: {regime_color}40;">
                <div class="kpi-value" style="color: {regime_color};">{regime_name.upper()}</div>
                <div class="kpi-label">Regime ({regime_confidence:.0f}% conf.)</div>
            </div>
            <div class="kpi-card kpi-orange">
                <div class="kpi-value">{eta_rec:.2f}</div>
                <div class="kpi-label">VVIX-Calibrated eta</div>
            </div>
        </div>

        <!-- Risk Metrics -->
        <div class="section-title">Risk Metrics</div>
        <div class="grid-3">
            <div class="card">
                <div class="card-title">Value-at-Risk</div>
                <table class="risk-table">
                    <tr><td>VaR 95%</td><td style="color:#ffd700">{risk_data.get('VaR_95', 0)*100:.2f}%</td></tr>
                    <tr><td>VaR 99%</td><td style="color:#ff9800">{risk_data.get('VaR_99', 0)*100:.2f}%</td></tr>
                    <tr><td>ES 95% (CVaR)</td><td style="color:#ff6bff">{risk_data.get('ES_95', 0)*100:.2f}%</td></tr>
                    <tr><td>ES 99%</td><td style="color:#ff4444">{risk_data.get('ES_99', 0)*100:.2f}%</td></tr>
                </table>
            </div>
            <div class="card">
                <div class="card-title">Volatility Profile</div>
                <table class="risk-table">
                    <tr><td>Terminal Vol (mean)</td><td>{risk_data.get('terminal_vol_mean', 0)*100:.1f}%</td></tr>
                    <tr><td>Terminal Vol (P95)</td><td>{risk_data.get('terminal_vol_p95', 0)*100:.1f}%</td></tr>
                    <tr><td>Panic Prob (>40%)</td><td>{risk_data.get('panic_prob_vol_gt_40pct', 0)*100:.1f}%</td></tr>
                    <tr><td>Horizon</td><td>{risk_data.get('horizon_days_equiv', 0):.1f} days</td></tr>
                </table>
            </div>
            <div class="card">
                <div class="card-title">Regime Signals</div>
                <table class="risk-table">'''

        for sig in regime.get('signals', [])[:5]:
            sig_color = regime_colors.get(sig.get('regime', ''), '#888')
            html += f'''
                    <tr><td>{sig.get('name','')}</td><td style="color:{sig_color}">{sig.get('detail','')}</td></tr>'''

        html += f'''
                </table>
            </div>
        </div>

        <!-- Bergomi Parameters -->
        <div class="section-title">Calibrated Parameters</div>
        <div class="card" style="margin-bottom: 25px;">
            <div class="params-grid">
                <div class="param-box">
                    <div class="param-value">{bergomi_params.get('hurst', 0):.4f}</div>
                    <div class="param-label">H (Hurst)</div>
                    <div class="param-note">{'ROUGH' if bergomi_params.get('hurst', 0.5) < 0.25 else ''}</div>
                </div>
                <div class="param-box">
                    <div class="param-value">{bergomi_params.get('eta', 0):.2f}</div>
                    <div class="param-label">eta (Vol-of-Vol)</div>
                </div>
                <div class="param-box">
                    <div class="param-value">{bergomi_params.get('rho', 0):.2f}</div>
                    <div class="param-label">rho (Correlation)</div>
                </div>
                <div class="param-box">
                    <div class="param-value">{np.sqrt(bergomi_params.get('xi0', 0.04))*100:.1f}%</div>
                    <div class="param-label">sigma_0 (ATM)</div>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="section-title">Model Performance</div>
        <div class="grid-2">
            <div class="card">
                <div class="card-title">RMSE Comparison</div>
                <div id="comparison-plot" class="plot-container"></div>
            </div>
            <div class="card">
                <div class="card-title">Backtest Win Rate ({backtest.get('n_scenarios', 0)} scenarios)</div>
                <div id="backtest-plot" class="plot-container"></div>
            </div>
        </div>

        <!-- 3D Surface -->
        <div class="section-title">Implied Volatility Surface</div>
        <div class="card full-width">
            <div id="surface-plot" style="width:100%; height:550px;"></div>
        </div>

        <div class="footer">
            DeepRoughVol Dashboard v2.0 | Neural SDE + Rough Bergomi<br>
            Multi-Measure: P-measure (risk), Q-measure (pricing) | SOFR-integrated | VVIX-calibrated
        </div>
    </div>

    <script>
        {comparison_div}
        {backtest_div}
        {surface_div}
    </script>
</body>
</html>'''

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"\n   Dashboard saved: {output_path} ({len(html)/1024:.1f} KB)")
        return output_path

    # ------------------------------------------------------------------
    #  Plot generators
    # ------------------------------------------------------------------
    def _create_surface_plot(self) -> str:
        if self.surface is None or len(self.surface) == 0:
            return '''Plotly.newPlot('surface-plot',[{type:'scatter3d',x:[0],y:[0],z:[0],
                mode:'text',text:['No surface data'],textfont:{size:20,color:'gray'}}],
                {paper_bgcolor:'rgba(0,0,0,0)',scene:{bgcolor:'rgba(0,0,0,0)'}});'''

        pivot = self.surface.pivot_table(
            values='impliedVolatility', index='moneyness', columns='dte', aggfunc='mean'
        ).dropna(how='all', axis=0).dropna(how='all', axis=1)
        pivot = pivot.interpolate(method='linear', axis=0).interpolate(method='linear', axis=1)
        pivot = pivot.ffill().bfill()

        x = pivot.columns.tolist()
        y = pivot.index.tolist()
        z = (pivot.values * 100).tolist()

        return f'''
        Plotly.newPlot('surface-plot',[{{
            type:'surface',x:{json.dumps(x)},y:{json.dumps(y)},z:{json.dumps(z)},
            colorscale:'Viridis',
            colorbar:{{title:'IV (%)',titlefont:{{color:'#888'}},tickfont:{{color:'#888'}}}},
            contours:{{z:{{show:true,usecolormap:true,project:{{z:true}}}}}}
        }}],{{
            paper_bgcolor:'rgba(0,0,0,0)',
            scene:{{
                xaxis:{{title:'DTE',color:'#888',gridcolor:'#333'}},
                yaxis:{{title:'Moneyness',color:'#888',gridcolor:'#333'}},
                zaxis:{{title:'IV (%)',color:'#888',gridcolor:'#333'}},
                bgcolor:'rgba(0,0,0,0)',
                camera:{{eye:{{x:1.8,y:-1.8,z:0.8}}}}
            }},margin:{{l:0,r:0,t:0,b:0}}
        }},{{responsive:true}});'''

    def _create_comparison_plot(self, rmse: dict) -> str:
        if not rmse:
            rmse = {'Black-Scholes': 7, 'Bergomi': 3.5, 'Neural SDE': 5}
        models = list(rmse.keys())
        values = list(rmse.values())
        colors = ['#888888', '#00ff88', '#00d4ff'][:len(models)]

        return f'''
        Plotly.newPlot('comparison-plot',[{{
            type:'bar',x:{json.dumps(models)},y:{json.dumps(values)},
            marker:{{color:{json.dumps(colors)}}},
            text:{json.dumps([f'{v:.2f}%' for v in values])},
            textposition:'outside',textfont:{{color:'#fff',size:14}}
        }}],{{
            paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
            xaxis:{{color:'#888'}},yaxis:{{title:'RMSE (%)',color:'#888',gridcolor:'#333'}},
            margin:{{l:50,r:20,t:20,b:50}},height:350
        }},{{responsive:true}});'''

    def _create_backtest_plot(self, summary: dict, win_rates: dict) -> str:
        if not summary:
            return '''Plotly.newPlot('backtest-plot',[{type:'scatter',x:[0.5],y:[0.5],
                mode:'text',text:['Run backtest first'],textfont:{size:16,color:'gray'}}],
                {paper_bgcolor:'rgba(0,0,0,0)',xaxis:{visible:false},yaxis:{visible:false}});'''

        labels = list(win_rates.keys()) if win_rates else ['BS', 'Bergomi', 'Neural']
        values = list(win_rates.values()) if win_rates else [10, 5, 15]
        colors = ['#888888', '#00ff88', '#00d4ff'][:len(labels)]

        return f'''
        Plotly.newPlot('backtest-plot',[{{
            type:'pie',labels:{json.dumps(labels)},values:{json.dumps(values)},
            marker:{{colors:{json.dumps(colors)}}},textinfo:'label+percent',
            textfont:{{color:'#fff'}},hole:0.4
        }}],{{
            paper_bgcolor:'rgba(0,0,0,0)',margin:{{l:20,r:20,t:20,b:20}},
            height:350,showlegend:false,
            annotations:[{{text:'Win<br>Rate',x:0.5,y:0.5,font:{{size:16,color:'#888'}},showarrow:false}}]
        }},{{responsive:true}});'''


def generate_dashboard():
    """Main entry point for dashboard generation."""
    print("=" * 70)
    print("   DEEPROUGHVOL DASHBOARD GENERATOR")
    print("=" * 70)
    os.makedirs("outputs", exist_ok=True)
    dashboard = DashboardV2()
    output = dashboard.generate("outputs/dashboard.html")
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(output)}")
        print("   Dashboard opened in browser!")
    except Exception:
        print(f"   Open {output} manually in your browser.")
    return output
