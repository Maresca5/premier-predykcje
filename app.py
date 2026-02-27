import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import plotly.graph_objects as go
import plotly.express as px

# ===========================================================================
# KOMPONENTY KOMERCYJNE - "EDGE VISUALIZATION"
# ===========================================================================

class EdgeVisualizer:
    """
    Wizualizuje przewagę matematyczną w sposób, który użytkownik 
    rozumie jako "szansę na zysk".
    """
    
    def render_money_on_table(self, combo: Dict, stake: float = 100.0):
        """
        Główny komponent: pokazuje Expected Profit zamiast abstrakcyjnego EV.
        """
        ev = combo.get('ev', 0)
        prob = combo.get('combined_prob', 0)
        odds = combo.get('combined_odds', 2.0)
        
        expected_profit = stake * ev
        break_even_rate = 1 / odds
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Expected Profit - to rozumieją wszyscy
            profit_color = "#4CAF50" if expected_profit > 0 else "#F44336"
            fig = go.Figure(go.Indicator(
                mode = "number+delta",
                value = expected_profit,
                number = {'prefix': "€", 'font': {'size': 48, 'color': profit_color}},
                delta = {'reference': 0, 'relative': False, 'valueformat': '.0f'},
                title = {'text': "Expected Profit<br><span style='font-size:0.5em'>na 100€ stawki</span>"},
                domain = {'row': 0, 'column': 0}
            ))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Prawdopodobieństwo vs Break-even
            fig = go.Figure()
            
            # Pasek: prawdziwe prawdopodobieństwo
            fig.add_trace(go.Bar(
                x=[prob * 100],
                y=['Model'],
                orientation='h',
                name='Model P%',
                marker_color='#4CAF50',
                text=f"{prob:.1%}",
                textposition='inside'
            ))
            
            # Linia: break-even
            fig.add_trace(go.Scatter(
                x=[break_even_rate * 100],
                y=['Model'],
                mode='markers+text',
                name='Break-even',
                marker=dict(size=20, color='#F44336', symbol='line-ns'),
                text=['Break-even'],
                textposition='top center'
            ))
            
            fig.update_layout(
                title="Edge Visualization",
                xaxis_title="Prawdopodobieństwo (%)",
                showlegend=False,
                height=200,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Symulacja Monte Carlo - "co jeśli"
            simulations = np.random.binomial(1, prob, 1000)
            cumulative = np.cumsum(simulations * odds * stake - stake)
            
            fig = px.line(
                x=range(1, 1001),
                y=cumulative,
                title="Symulacja: 1000 zakładów",
                labels={'x': 'Numer zakładu', 'y': 'Zysk/Strata (€)'}
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    def render_confidence_interval(self, combo: Dict):
        """
        Pokazuje niepewność jako "strefę ryzyka".
        """
        prob = combo.get('combined_prob', 0.5)
        n_samples = combo.get('sample_size', 100)
        
        # Wilson score interval
        z = 1.96
        denominator = 1 + z**2/n_samples
        centre_adjusted_probability = prob + z*z / (2*n_samples)
        adjusted_standard_deviation = np.sqrt((prob*(1-prob) + z*z / (4*n_samples)) / n_samples)
        
        lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
        upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
        
        fig = go.Figure()
        
        # Strefa zysku
        fig.add_vrect(
            x0=lower_bound, x1=upper_bound,
            fillcolor="LightSalmon", opacity=0.3,
            layer="below", line_width=0,
        )
        
        # Punkt estymaty
        fig.add_trace(go.Scatter(
            x=[prob],
            y=[1],
            mode='markers',
            marker=dict(size=30, color='#2196F3'),
            name='Model Estimate'
        ))
        
        # Break-even line
        be = 1 / combo.get('combined_odds', 2.0)
        fig.add_vline(x=be, line_dash="dash", line_color="red", 
                     annotation_text="Break-even")
        
        fig.update_layout(
            title=f"Niepewność modelu (95% CI)<br>"
                  f"Prawdopodobieństwo: {lower_bound:.1%} - {upper_bound:.1%}",
            xaxis_title="Prawdopodobieństwo sukcesu",
            yaxis_visible=False,
            showlegend=False,
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretacja
        if lower_bound > be:
            st.success("✅ **High Confidence Edge** - Nawet pesymistyczny scenariusz daje zysk")
        elif upper_bound < be:
            st.error("❌ **No Edge** - Nawet optymistyczny scenariusz poniżej break-even")
        else:
            st.warning("⚠️ **Marginal Edge** - Zależy od wariancji, wymaga większej próby")

class MarketDepthAnalyzer:
    """
    Analizuje "głębokość" rynku - jak wiele niezależnych źródeł potwierdza value.
    """
    
    def __init__(self):
        self.signal_sources = {
            'model_price': 0.4,
            'closing_line': 0.3,
            'sharp_money': 0.2,
            'market_sentiment': 0.1
        }
    
    def calculate_conviction_score(self, signals: Dict) -> Dict:
        """
        Agregacja wielu sygnałów w "Conviction Score".
        """
        weighted_sum = 0
        total_weight = 0
        
        for source, weight in self.signal_sources.items():
            if source in signals:
                weighted_sum += signals[source] * weight
                total_weight += weight
        
        if total_weight == 0:
            return {'score': 0, 'confidence': 'none'}
        
        normalized_score = weighted_sum / total_weight
        
        values = list(signals.values())
        dispersion = np.std(values) if len(values) > 1 else 0.5
        
        agreement_bonus = (1 - dispersion) * 0.2
        
        final_score = min(normalized_score + agreement_bonus, 1.0)
        
        tiers = {
            (0.8, 1.0): ('Maximum Conviction', '#4CAF50', 'Wszystkie sygnały zgodne'),
            (0.6, 0.8): ('Strong Conviction', '#8BC34A', 'Większość sygnałów potwierdza'),
            (0.4, 0.6): ('Moderate Conviction', '#FF9800', 'Mieszane sygnały, wymaga ostrożności'),
            (0.0, 0.4): ('Low Conviction', '#F44336', 'Brak zgodności, unikać')
        }
        
        for (low, high), (label, color, desc) in tiers.items():
            if low <= final_score < high:
                return {
                    'score': final_score,
                    'label': label,
                    'color': color,
                    'description': desc,
                    'dispersion': dispersion
                }
        
        return {'score': final_score, 'label': 'Unknown', 'color': '#888', 'description': 'N/A'}

# ===========================================================================
# INNOWACJE KOMERCYJNE - NOWE FUNKCJONALNOŚCI
# ===========================================================================

class SyndicateTools:
    """
    Narzędzia dla "poważnych" graczy - syndykatów i profesjonalistów.
    """
    
    def render_staking_calculator(self, edge: float, bankroll: float, 
                                  kelly_fraction: float = 0.25):
        """
        Zaawansowany kalkulator stawek z Kelly Criterion.
        """
        b = edge
        p = 0.5 + edge/2
        q = 1 - p
        
        kelly_pct = (b * p - q) / b if b > 0 else 0
        kelly_pct = max(0, min(kelly_pct, 0.5))
        
        fractional_kelly = kelly_pct * kelly_fraction
        
        recommended_stake = bankroll * fractional_kelly
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Full Kelly",
                f"{kelly_pct:.1%}",
                help="Teoretycznie optymalna stawka - ale zbyt agresywna w praktyce"
            )
        
        with col2:
            st.metric(
                f"Fractional Kelly ({kelly_fraction:.0%})",
                f"{fractional_kelly:.1%}",
                help="Konserwatywna wersja Kelly - zalecana dla większości graczy"
            )
        
        with col3:
            st.metric(
                "Rekomendowana stawka",
                f"€{recommended_stake:.2f}",
                help=f"Przy bankrollu €{bankroll:.0f}"
            )
        
        fractions = np.linspace(0, kelly_pct * 1.5, 50)
        growth_rates = [self._expected_growth_rate(f, p, q, b) for f in fractions]
        
        fig = px.line(
            x=fractions * 100,
            y=growth_rates,
            title="Expected Growth Rate vs Stake Size",
            labels={'x': 'Stawka (% bankrollu)', 'y': 'Expected log growth'}
        )
        fig.add_vline(x=kelly_pct*100, line_dash="dash", line_color="green",
                     annotation_text="Full Kelly")
        fig.add_vline(x=fractional_kelly*100, line_dash="dash", line_color="orange",
                     annotation_text="Your setting")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _expected_growth_rate(self, f: float, p: float, q: float, b: float) -> float:
        """Expected logarithmic growth rate"""
        return p * np.log(1 + f * b) + q * np.log(1 - f)
    
    def render_portfolio_optimizer(self, opportunities: List[Dict]):
        """
        Optymalizacja portfela zakładów - unikanie overexposure.
        """
        if len(opportunities) < 2:
            st.info("Dodaj więcej zakładów, aby zobaczyć optymalizację portfela")
            return
        
        corr_matrix = self._estimate_correlation_matrix(opportunities)
        
        st.markdown("### 📊 Optymalizacja Portfela")
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(x="Zakład", y="Zakład", color="Korelacja"),
            x=[f"Z{i+1}" for i in range(len(opportunities))],
            y=[f"Z{i+1}" for i in range(len(opportunities))],
            color_continuous_scale="RdBu",
            aspect="equal"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        high_corr_pairs = []
        for i in range(len(opportunities)):
            for j in range(i+1, len(opportunities)):
                if corr_matrix[i, j] > 0.7:
                    high_corr_pairs.append((i, j, corr_matrix[i, j]))
        
        if high_corr_pairs:
            st.warning("⚠️ Wykryto wysoko skorelowane zakłady - rozważ redukcję stawek:")
            for i, j, corr in high_corr_pairs:
                st.markdown(f"• Zakład {i+1} i {j+1}: {corr:.1%} korelacji")
                st.markdown(f"  - {opportunities[i]['name'][:50]}...")
                st.markdown(f"  - {opportunities[j]['name'][:50]}...")
    
    def _estimate_correlation_matrix(self, opportunities: List[Dict]) -> np.ndarray:
        """Estymuje macierz korelacji na podstawie typów rynków"""
        n = len(opportunities)
        corr = np.eye(n)
        
        for i in range(n):
            for j in range(i+1, n):
                m1 = opportunities[i].get('market_type', 'unknown')
                m2 = opportunities[j].get('market_type', 'unknown')
                
                base_corr = 0.3 if m1 == m2 else 0.1
                
                same_match = opportunities[i].get('match') == opportunities[j].get('match')
                if same_match:
                    base_corr += 0.3
                
                corr[i, j] = corr[j, i] = min(base_corr, 0.95)
        
        return corr

# ===========================================================================
# UI INTEGRATION
# ===========================================================================

def render_commercial_interface(combo: Dict, bankroll: float = 1000.0):
    """
    Główny interfejs komercyjny dla pojedynczego combo.
    """
    visualizer = EdgeVisualizer()
    syndicate = SyndicateTools()
    
    st.markdown("---")
    st.markdown("## 💰 Analiza Komercyjna")
    
    visualizer.render_money_on_table(combo)
    
    with st.expander("📊 Szczegółowa analiza ryzyka", expanded=False):
        visualizer.render_confidence_interval(combo)
    
    with st.expander("🎯 Kalkulator Stawek (Kelly Criterion)", expanded=True):
        kelly_frac = st.slider("Kelly Fraction", 0.05, 0.5, 0.25, 0.05,
                              help="Ułamek Kelly - zalecane 0.25 dla początkujących, 0.5 dla zaawansowanych")
        syndicate.render_staking_calculator(combo.get('ev', 0), bankroll, kelly_frac)
    
    st.session_state.setdefault('portfolio', [])
    if st.button("➕ Dodaj do portfela"):
        st.session_state['portfolio'].append(combo)
        st.success("Dodano! Przejdź do zakładki 'Portfel' aby zobaczyć optymalizację.")

# ===========================================================================
# PRZYKŁAD UŻYCIA
# ===========================================================================

def example_usage():
    st.set_page_config(page_title="Smart Bet Builder Pro", layout="wide")
    
    # POPRAWIONY SŁOWNIK - brakowało zamknięcia klamry w 'legs'
    combo_example = {
        'type': 'double',
        'legs': [
            {'name': 'Over 2.5 goals', 'probability': 0.62, 'market_type': 'goals_over'},
            {'name': 'BTTS Yes', 'probability': 0.58, 'market_type': 'btts_yes'}
        ],  # <-- TO BYŁO BRAKUJĄCE W TWOIM KODZIE
        'combined_prob': 0.41,
        'naive_prob': 0.36,
        'correlation_effect': 0.14,
        'ev': 0.23,
        'combined_odds': 3.2,
        'rationale': 'Positive correlation play: open match expected'
    }
    
    render_commercial_interface(combo_example, bankroll=2000.0)
    
    if st.session_state.get('portfolio'):
        st.markdown("---")
        st.markdown("## 📁 Twój Portfel")
        syndicate = SyndicateTools()
        syndicate.render_portfolio_optimizer(st.session_state['portfolio'])

if __name__ == "__main__":
    example_usage()
