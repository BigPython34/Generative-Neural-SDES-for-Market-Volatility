"""
Interactive Dashboard
=====================
Generates a standalone HTML dashboard with all calibration results.
Can be shared and viewed without Python installation.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DashboardGenerator:
    """
    Generates an interactive HTML dashboard with all results.
    """
    
    def __init__(self, title: str = "Volatility Calibration Dashboard"):
        self.title = title
        self.figures = []
        self.metrics = {}
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
    def add_figure(self, fig: go.Figure, section: str):
        """Add a Plotly figure to the dashboard."""
        self.figures.append({'figure': fig, 'section': section})
    
    def add_metrics(self, metrics: dict):
        """Add key metrics to display."""
        self.metrics.update(metrics)
    
    def load_calibration_results(self):
        """Load all available calibration results."""
        
        # Load calibration report
        if os.path.exists('data/calibration_report.json'):
            with open('data/calibration_report.json', 'r') as f:
                calib = json.load(f)
            self.metrics['calibration'] = calib
            print("Loaded calibration report")
        
        # Load backtest results
        if os.path.exists('data/backtest_results.json'):
            with open('data/backtest_results.json', 'r') as f:
                backtest = json.load(f)
            self.metrics['backtest'] = backtest
            print("Loaded backtest results")
        
        # Load surface data
        from quant.options_cache import OptionsDataCache
        cache = OptionsDataCache()
        try:
            surface, info = cache.load_latest("SPY")
            self.metrics['surface'] = {
                'n_options': len(surface),
                'maturities': sorted(surface['dte'].unique().tolist()),
                'spot': info['spot'],
                'snapshot_date': info['datetime']
            }
            self.surface = surface
            print("Loaded cached surface")
        except:
            self.surface = None
            print("No cached surface found")
    
    def create_summary_metrics(self) -> go.Figure:
        """Create KPI cards figure."""
        
        fig = go.Figure()
        
        # Create a grid of KPI-like annotations
        kpis = []
        
        if 'calibration' in self.metrics:
            calib = self.metrics['calibration']
            kpis.append(('Spot Price', f"${calib.get('spot', 0):.2f}", 'white'))
            kpis.append(('ATM IV', f"{calib.get('atm_iv', 0)*100:.1f}%", 'cyan'))
            kpis.append(('Options Loaded', str(calib.get('n_options', 0)), 'lime'))
            
            if 'rmse_results' in calib:
                best = min(calib['rmse_results'].items(), key=lambda x: x[1])
                kpis.append(('Best Model', best[0], 'gold'))
                kpis.append(('Best RMSE', f"{best[1]:.2f}%", 'gold'))
        
        if 'backtest' in self.metrics:
            bt = self.metrics['backtest']
            neural_rmse = bt['summary'].get('neural_sde_mean_rmse', 0)
            kpis.append(('Backtest Days', str(bt.get('n_days', 0)), 'magenta'))
            kpis.append(('Avg Neural RMSE', f"{neural_rmse:.2f}%", 'cyan'))
        
        # Layout KPIs in grid
        n_cols = 4
        n_rows = (len(kpis) + n_cols - 1) // n_cols
        
        for i, (label, value, color) in enumerate(kpis):
            row = i // n_cols
            col = i % n_cols
            
            x = (col + 0.5) / n_cols
            y = 1 - (row + 0.5) / max(n_rows, 1)
            
            # Value
            fig.add_annotation(
                x=x, y=y + 0.1,
                text=f"<b>{value}</b>",
                font=dict(size=28, color=color),
                showarrow=False,
                xref='paper', yref='paper'
            )
            
            # Label
            fig.add_annotation(
                x=x, y=y - 0.1,
                text=label,
                font=dict(size=14, color='gray'),
                showarrow=False,
                xref='paper', yref='paper'
            )
        
        fig.update_layout(
            title=dict(text='Key Metrics', font=dict(size=18, color='white')),
            template='plotly_dark',
            height=200,
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return fig
    
    def create_model_comparison(self) -> go.Figure:
        """Create model comparison bar chart."""
        
        fig = go.Figure()
        
        if 'calibration' in self.metrics and 'rmse_results' in self.metrics['calibration']:
            rmse = self.metrics['calibration']['rmse_results']
            
            models = list(rmse.keys())
            values = list(rmse.values())
            colors = ['gray', 'lime', 'cyan'][:len(models)]
            
            fig.add_trace(go.Bar(
                x=models,
                y=values,
                marker_color=colors,
                text=[f'{v:.2f}%' for v in values],
                textposition='outside',
                textfont=dict(size=16)
            ))
            
            # Add best marker
            best_idx = np.argmin(values)
            fig.add_annotation(
                x=models[best_idx],
                y=values[best_idx] + 1,
                text="BEST",
                showarrow=False,
                font=dict(size=14, color='gold')
            )
        
        fig.update_layout(
            title=dict(text='Model RMSE Comparison (30 DTE)', font=dict(size=16)),
            template='plotly_dark',
            height=350,
            yaxis_title='RMSE (%)',
            showlegend=False
        )
        
        return fig
    
    def create_bergomi_params(self) -> go.Figure:
        """Visualize Bergomi parameters."""
        
        fig = go.Figure()
        
        if 'calibration' in self.metrics and 'bergomi_params' in self.metrics['calibration']:
            params = self.metrics['calibration']['bergomi_params']
            
            param_list = [
                ('H (Hurst)', params.get('hurst', 0.1), 'ROUGH!' if params.get('hurst', 0.5) < 0.25 else ''),
                ('η (Vol-of-Vol)', params.get('eta', 2), ''),
                ('ρ (Correlation)', params.get('rho', -0.7), ''),
                ('ξ₀ (ATM Var)', params.get('xi0', 0.04), f"σ={np.sqrt(params.get('xi0', 0.04))*100:.0f}%")
            ]
            
            names = [p[0] for p in param_list]
            values = [p[1] for p in param_list]
            notes = [p[2] for p in param_list]
            
            fig.add_trace(go.Bar(
                x=names,
                y=values,
                marker_color=['cyan', 'lime', 'magenta', 'gold'],
                text=[f'{v:.3f}<br>{n}' for v, n in zip(values, notes)],
                textposition='outside',
                textfont=dict(size=14)
            ))
        
        fig.update_layout(
            title=dict(text='Calibrated Bergomi Parameters', font=dict(size=16)),
            template='plotly_dark',
            height=300,
            yaxis_title='Value'
        )
        
        return fig
    
    def create_surface_3d(self) -> go.Figure:
        """Create 3D volatility surface."""
        
        if self.surface is None:
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5, text="No surface data available",
                showarrow=False, font=dict(size=20, color='gray'),
                xref='paper', yref='paper'
            )
            fig.update_layout(template='plotly_dark', height=500)
            return fig
        
        # Pivot to grid
        pivot = self.surface.pivot_table(
            values='impliedVolatility', 
            index='moneyness', 
            columns='dte', 
            aggfunc='mean'
        )
        
        X = pivot.columns.values
        Y = pivot.index.values
        Z = pivot.values * 100
        
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            colorbar=dict(title='IV (%)')
        )])
        
        spot = self.metrics.get('surface', {}).get('spot', 0)
        
        fig.update_layout(
            title=dict(text=f'SPY Implied Volatility Surface (Spot: ${spot:.2f})', font=dict(size=16)),
            scene=dict(
                xaxis_title='Days to Expiry',
                yaxis_title='Log-Moneyness',
                zaxis_title='Implied Vol (%)',
                camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8))
            ),
            template='plotly_dark',
            height=500
        )
        
        return fig
    
    def create_backtest_summary(self) -> go.Figure:
        """Create backtest summary visualization."""
        
        if 'backtest' not in self.metrics:
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5, text="No backtest data. Run backtesting.py first.",
                showarrow=False, font=dict(size=16, color='gray'),
                xref='paper', yref='paper'
            )
            fig.update_layout(template='plotly_dark', height=300)
            return fig
        
        bt = self.metrics['backtest']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Average RMSE by Model', 'Win Rate'),
            specs=[[{}, {"type": "pie"}]]
        )
        
        # RMSE comparison
        summary = bt['summary']
        models = ['Black-Scholes', 'Bergomi', 'Neural SDE']
        rmse_values = [
            summary.get('bs_mean_rmse', 0),
            summary.get('bergomi_mean_rmse', 0),
            summary.get('neural_sde_mean_rmse', 0)
        ]
        
        fig.add_trace(go.Bar(
            x=models,
            y=rmse_values,
            marker_color=['gray', 'lime', 'cyan'],
            text=[f'{v:.2f}%' for v in rmse_values],
            textposition='outside'
        ), row=1, col=1)
        
        # Win rate pie
        win_rates = bt.get('win_rates', {})
        if win_rates:
            fig.add_trace(go.Pie(
                labels=list(win_rates.keys()),
                values=list(win_rates.values()),
                marker_colors=['gray', 'lime', 'cyan'],
                textinfo='label+percent',
                hole=0.4
            ), row=1, col=2)
        
        fig.update_layout(
            title=dict(text=f'Backtest Results ({bt.get("n_days", 0)} days)', font=dict(size=16)),
            template='plotly_dark',
            height=350,
            showlegend=False
        )
        
        return fig
    
    def generate_html(self, output_path: str = "outputs/dashboard.html"):
        """Generate standalone HTML dashboard."""
        
        print("\nGenerating Interactive Dashboard...")
        print("-" * 50)
        
        # Load data
        self.load_calibration_results()
        
        # Create all figures
        figs = [
            ('metrics', self.create_summary_metrics()),
            ('comparison', self.create_model_comparison()),
            ('bergomi', self.create_bergomi_params()),
            ('surface', self.create_surface_3d()),
            ('backtest', self.create_backtest_summary())
        ]
        
        # Build HTML
        html_parts = [
            f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{self.title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            background: #1a1a2e;
            color: #eee;
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        h1 {{
            color: #00d4ff;
            margin: 0;
        }}
        .subtitle {{
            color: #888;
            margin-top: 5px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        .card {{
            background: #16213e;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.title}</h1>
        <div class="subtitle">Generated: {self.timestamp}</div>
    </div>
    <div class="grid">
'''
        ]
        
        # Add figures
        layouts = {
            'metrics': 'full-width',
            'comparison': '',
            'bergomi': '',
            'surface': 'full-width',
            'backtest': 'full-width'
        }
        
        for name, fig in figs:
            layout_class = layouts.get(name, '')
            fig_html = pio.to_html(fig, full_html=False, include_plotlyjs=False)
            
            html_parts.append(f'''
        <div class="card {layout_class}">
            {fig_html}
        </div>
''')
        
        # Close HTML
        html_parts.append(f'''
    </div>
    <div class="footer">
        Neural SDE Volatility Calibration | Rough Bergomi Benchmark<br>
        Data source: SPY Options via Yahoo Finance | Generated with Plotly
    </div>
</body>
</html>
''')
        
        # Write file
        html_content = ''.join(html_parts)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Dashboard saved to: {output_path}")
        print(f"   Size: {len(html_content) / 1024:.1f} KB")
        print(f"\nOpen in browser to view!")
        
        return output_path


def generate_dashboard():
    """Main entry point for dashboard generation."""
    
    print("="*70)
    print("   INTERACTIVE DASHBOARD GENERATOR")
    print("="*70)
    
    generator = DashboardGenerator("Neural SDE Volatility Calibration")
    output = generator.generate_html("dashboard.html")
    
    # Try to open in browser
    try:
        import webbrowser
        webbrowser.open(f'file://{os.path.abspath(output)}')
        print("   Dashboard opened in browser!")
    except:
        print(f"   Open {output} manually in your browser.")
    
    return output


if __name__ == "__main__":
    generate_dashboard()
