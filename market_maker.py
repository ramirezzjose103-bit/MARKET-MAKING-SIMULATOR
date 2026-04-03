"""

MARKET MAKING SIMULATOR — Modelo Avellaneda-Stoikov         
Activo real: USD/MXN  (datos vía yfinance)                  
                                                                
Flujo:                                                         
    1. Descarga precios históricos reales de USD/MXN             
    2. Calibra σ con la volatilidad histórica del par            
    3. Corre el MM sobre esos precios reales 
    4. Calcula PnL con                       
       a) Bruto (sin costos)                                                   
       b) ψ · Turnover (modelo industria)                
                                                                 
  Fórmulas A-S:                                                  
    r*(t) = s(t) - q·γ·σ²·(T-t)    ← precio de reserva           
    δ*(t) = γ·σ²·(T-t) + 2/γ·ln(1+γ/κ)  ← spread óptimo          
    bid   = r* - δ*/2                                            
    ask   = r* + δ*/2                                            
             

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from dataclasses import dataclass
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Instala yfinance:  pip install yfinance")



# ══════════════════════════════════════════════════════════════════
# 1. DESCARGA Y CALIBRACIÓN CON DATOS REALES
# ══════════════════════════════════════════════════════════════════

def download_fx(ticker: str = "USDMXN=X",
                period: str = "5d",
                interval: str = "1m") -> pd.DataFrame:
    """
    Descarga datos FX de Yahoo Finance.

    Args:
        ticker   : par FX en formato Yahoo  (ej. 'USDMXN=X', 'EURUSD=X')
        period   : período a descargar      ('1d','5d','1mo','3mo','1y')
        interval : granularidad de las velas ('1m','5m','15m','30m','1h','1d')
                   ⚠ Datos de 1m solo disponibles para los últimos 7 días.

    Returns:
        DataFrame con columnas [Open, High, Low, Close] indexado por datetime.
    """
    print(f"  ▶ Descargando {ticker}  período={period}  intervalo={interval} ...")
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(
            f"No se obtuvieron datos para '{ticker}'.\n"
            "  Verifica el ticker y tu conexión a internet.\n"
            "  Ejemplos válidos: 'USDMXN=X', 'EURUSD=X', 'GBPUSD=X'"
        )

    # Aplanar columnas multi-nivel (yfinance >= 0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close']].dropna()

    print(f"  ✓ {len(df)} velas descargadas")
    print(f"    Desde : {df.index[0].strftime('%Y-%m-%d %H:%M')}")
    print(f"    Hasta : {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"    Rango : {df['Close'].min():.4f} – {df['Close'].max():.4f} MXN/USD")
    return df


def calibrate_sigma(prices: np.ndarray, dt_seconds: float = 60.0) -> float:
    """
    Estima σ por VELA a partir de retornos log.
    Proceso:
        r_i = ln(S_i / S_{i-1})
        σ_vela = std(r_i)            
        σ_anual = σ_vela × √(252 × barras/día) 
    """
    log_ret       = np.diff(np.log(prices))
    sigma_per_bar = np.std(log_ret)
    bars_per_day  = (6.5 * 3600) / dt_seconds
    sigma_annual  = sigma_per_bar * np.sqrt(252 * bars_per_day)

    print(f"  Volatilidad calibrada:")
    print(f"      σ por vela ({int(dt_seconds//60)}m) = {sigma_per_bar:.6f}")
    print(f"      σ anualizada       = {sigma_annual*100:.2f}%")
    print(f"      (modelo usa σ por vela con dt=1)")
    return sigma_per_bar


# ══════════════════════════════════════════════════════════════════
# 2. PARÁMETROS DEL MODELO
# ══════════════════════════════════════════════════════════════════

@dataclass
class ModelParams:
    """
    Parámetros del modelo A-S adaptados a FX (USD/MXN).

    Unidades:
      - Precios en MXN por USD  (ej. ~17.15)
      - PnL en MXN
      - Inventario en USD (unidades = 1 USD)
      - psi y fee_fixed en MXN
    """

    # Precio inicial — se sobreescribe con el primer precio real
    S0: float = 17.0

    # σ por VELA — calibrado automáticamente desde datos reales
    # Con velas de 1m para USD/MXN, σ_vela ≈ 0.0001 – 0.0003
    sigma: float = 0.0002

    # Aversión al riesgo (γ)
    # Valores bajos (0.01-0.1) → spreads angostos, más trades, más riesgo inventario
    # Valores altos (1-5)      → spreads anchos, menos trades, más conservador
    # Para USD/MXN con velas 1m: γ=0.1 da spreads de ~1-3 pips (realista)
    gamma: float = 0.1

    # Intensidad de llegada de órdenes (λ): órdenes por VELA
    # Con dt=1 vela de 1m, λ=0.3 → ~18 trades/hora (razonable para MM retail)
    lam: float = 0.3

    # Decaimiento de intensidad con distancia al mid (κ)
    # κ controla qué tan rápido cae la probabilidad de ejecución.
    # Con spreads de ~1-5 pips (0.0001-0.0005 MXN) y κ=5000:
    #   dist=0.0002 → intensidad × exp(-5000×0.0002) = × exp(-1) ≈ 37%
    # Regla: κ ≈ 1 / (spread_típico_en_MXN)
    kappa: float = 5000.0

    # Horizonte en VELAS (se sobreescribe con len(prices) en main)
    T: float = 390.0

    # Paso de tiempo en VELAS (dt=1 → 1 vela por paso)
    dt: float = 1.0

    # Límite de inventario en USD
    max_inventory: int = 100

    # ── Comisiones (fee por trade) ────────────────────────────────
    # Fee porcentual: 2 bps es típico de broker retail FX
    fee_rate: float = 0.0002

    # Fee fija por trade en MXN (0 en FX spot normalmente)
    fee_fixed: float = 0.0

    # ── Modelo ψ·Turnover  ─────────────────────────────────
    # ψ = half-spread del mercado externo en MXN/USD
    # Spread típico USD/MXN ≈ 0.001 MXN → half-spread = 0.0005
    psi: float = 0.0005

    seed: int = None


# ══════════════════════════════════════════════════════════════════
# 3. MODELO AVELLANEDA-STOIKOV
# ══════════════════════════════════════════════════════════════════

class AvellanedaStoikov:
    """Estrategia óptima de market making (Avellaneda & Stoikov, 2008)."""

    def __init__(self, params: ModelParams):
        self.p = params

    def reservation_price(self, s: float, q: float, t: float) -> float:
        """
        r*(t) = s(t) − q · γ · σ² · (T − t)

        Si q > 0 (long): r* < s  → el MM baja quotes para incentivar ventas.
        Si q < 0 (short): r* > s → el MM sube quotes para incentivar compras.
        """
        tau = max(self.p.T - t, 1e-6)
        return s - q * self.p.gamma * self.p.sigma**2 * tau

    def optimal_spread(self, t: float) -> float:
        """
        δ*(t) = γ·σ²·(T−t)  +  (2/γ)·ln(1 + γ/κ)

        Decrece con el tiempo porque queda menos horizonte para rebalancear.
        """
        tau = max(self.p.T - t, 1e-6)
        return (self.p.gamma * self.p.sigma**2 * tau
                + (2.0 / self.p.gamma) * np.log(1.0 + self.p.gamma / self.p.kappa))

    def compute_quotes(self, s: float, q: float,
                       t: float) -> Tuple[float, float, float, float]:
        """Devuelve (bid, ask, r*, δ*)."""
        r     = self.reservation_price(s, q, t)
        delta = self.optimal_spread(t)
        return r - delta / 2.0, r + delta / 2.0, r, delta

    def arrival_intensity(self, quote: float, mid: float) -> float:
        """Intensidad de Poisson: λ_eff = λ · exp(−κ · |mid − quote|)"""
        return self.p.lam * np.exp(-self.p.kappa * abs(mid - quote))


# ══════════════════════════════════════════════════════════════════
# 4. DATACLASSES
# ══════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    time_idx: int         # índice en el array de precios
    side: str             # 'BUY' o 'SELL'
    price: float          # precio de ejecución (MXN/USD)
    inventory_after: int  # inventario tras el trade


@dataclass
class SimResult:
    times: np.ndarray             # timestamps reales (datetime)
    mid_prices: np.ndarray        # precios Close reales (MXN/USD)
    bid_quotes: np.ndarray
    ask_quotes: np.ndarray
    reservation_prices: np.ndarray
    spreads: np.ndarray           # δ*(t)
    inventories: np.ndarray       # inventario neto en USD
    cash: np.ndarray              # caja en MXN
    pnl: np.ndarray               # PnL neto (fee por trade)
    turnover: np.ndarray          # Σ|Δq_t| acumulado (USD)
    pnl_turnover: np.ndarray      # PnL = Σ R_t − ψ·Σ|Δq_t|  
    gross_returns: np.ndarray     # Σ R_t bruto (antes de costo ψ)
    trades: List[Trade]
    params: ModelParams
    ticker: str
    sigma_calibrated: float


# ══════════════════════════════════════════════════════════════════
# 5. SIMULADOR SOBRE PRECIOS REALES
# ══════════════════════════════════════════════════════════════════

class MarketMakingSimulator:
    """
    Corre la estrategia A-S sobre una serie de precios reales de FX.

    Aquí los
    precios mid son los datos reales descargados de Yahoo Finance.
    El MM solo controla sus quotes bid/ask en cada tick.
    """

    def __init__(self, params: ModelParams):
        self.p     = params
        self.model = AvellanedaStoikov(params)
        if params.seed is not None:
            np.random.seed(params.seed)

    def simulate(self, prices: np.ndarray,
                 timestamps: np.ndarray) -> SimResult:
        """
        Args:
            prices     : array de precios mid reales (Close de yfinance)
            timestamps : array de datetimes correspondientes
        """
        p  = self.p
        n  = len(prices)

        # Inicializar arrays
        bid_quotes       = np.zeros(n)
        ask_quotes       = np.zeros(n)
        res_prices       = np.zeros(n)
        spreads          = np.zeros(n)
        inventories      = np.zeros(n, dtype=int)
        cash_arr         = np.zeros(n)
        pnl_arr          = np.zeros(n)
        turnover_arr     = np.zeros(n)
        pnl_turnover_arr = np.zeros(n)
        gross_ret_arr    = np.zeros(n)

        # Estado
        q              = 0
        C              = 0.0
        q_prev         = 0
        total_turnover = 0.0
        total_gross_R  = 0.0
        trades: List[Trade] = []

        for i in range(n):
            S = prices[i]
            t = i * p.dt   # tiempo relativo en segundos

            # ── 1. Quotes óptimos ───────────────────────────────────
            bid, ask, r, delta = self.model.compute_quotes(S, q, t)

            # ── 2. Llegada de órdenes (proceso de Poisson) ──────────
            lam_bid  = self.model.arrival_intensity(bid, S)
            lam_ask  = self.model.arrival_intensity(ask, S)
            bid_hit  = (np.random.random() < lam_bid  * p.dt) and (q <  p.max_inventory)
            ask_lift = (np.random.random() < lam_ask  * p.dt) and (q > -p.max_inventory)

            # ── 3. Ejecución de trades ──────────────────────────────
            if bid_hit:                         # alguien vende al MM
                q += 1
                C -= bid
                trades.append(Trade(i, 'BUY', bid, q))

            if ask_lift:                        # alguien compra al MM
                q -= 1
                C += ask
                trades.append(Trade(i, 'SELL', ask, q))

            # ── 4. Turnover y retorno bruto  ──────────
            # Turnover_t = |Δq_t|
            total_turnover += abs(q - q_prev)

            # R_t = q_{t-1} · ΔS  (ganancia/pérdida por movimiento de precio)
            S_prev = prices[i - 1] if i > 0 else S
            total_gross_R += q_prev * (S - S_prev)

            # PnL_ψ = Σ R_t − ψ · Σ|Δq_t|
            pnl_turnover = total_gross_R - p.psi * total_turnover
            q_prev = q

            # ── 5. PnL mark-to-market ───────────────────────────────
            pnl = C + q * S

            # ── 6. Guardar ──────────────────────────────────────────
            bid_quotes[i]       = bid
            ask_quotes[i]       = ask
            res_prices[i]       = r
            spreads[i]          = delta
            inventories[i]      = q
            cash_arr[i]         = C
            pnl_arr[i]          = pnl
            turnover_arr[i]     = total_turnover
            pnl_turnover_arr[i] = pnl_turnover
            gross_ret_arr[i]    = total_gross_R

        return SimResult(
            times=timestamps,
            mid_prices=prices,
            bid_quotes=bid_quotes,
            ask_quotes=ask_quotes,
            reservation_prices=res_prices,
            spreads=spreads,
            inventories=inventories,
            cash=cash_arr,
            pnl=pnl_arr,
            turnover=turnover_arr,
            pnl_turnover=pnl_turnover_arr,
            gross_returns=gross_ret_arr,
            trades=trades,
            params=p,
            ticker="USDMXN=X",
            sigma_calibrated=p.sigma,
        )


# ══════════════════════════════════════════════════════════════════
# 6. ESTADÍSTICAS
# ══════════════════════════════════════════════════════════════════

def compute_stats(result: SimResult) -> dict:
    """
    PnL = Σ R_t  −  ψ · Σ|Δq_t|
      Σ R_t    = suma de retornos brutos (ganancia de spread + inventario)
      Σ|Δq_t|  = turnover total (USD movidos en total)
      ψ        = costo por unidad movida (half-spread del mercado)
    """
    r = result
    p = r.params

    n_total  = len(r.trades)
    n_buys   = sum(1 for t in r.trades if t.side == 'BUY')
    n_sells  = sum(1 for t in r.trades if t.side == 'SELL')

    total_turnover    = r.turnover[-1]
    total_gross_R     = r.gross_returns[-1]
    pnl_psi           = r.pnl_turnover[-1]
    cost_psi          = p.psi * total_turnover
    turnover_drag_pct = (cost_psi / abs(total_gross_R) * 100) if total_gross_R != 0 else 0.0

    pnl_vol      = np.std(r.pnl_turnover)
    sharpe       = (pnl_psi / pnl_vol) if pnl_vol > 0 else 0.0
    max_drawdown = np.min(r.pnl_turnover - np.maximum.accumulate(r.pnl_turnover))
    avg_spread   = np.mean(r.spreads)
    max_inv      = int(np.max(np.abs(r.inventories)))
    price_ret    = (r.mid_prices[-1] / r.mid_prices[0] - 1) * 100

    return {
        'PnL ψ·Turnover (MXN)': f"${pnl_psi:.4f}",
        'Σ R_t bruto':          f"${total_gross_R:.4f}",
        'Σ|Δq| Turnover (USD)': f"{total_turnover:.0f}",
        'Costo ψ·Turnover':     f"${cost_psi:.5f}",
        'Arrastre Turnover':    f"{turnover_drag_pct:.1f}% del bruto",
        'Trades Totales':       n_total,
        'Buys / Sells':         f"{n_buys} / {n_sells}",
        'Max |Inventario|':     f"{max_inv} USD",
        'Spread Prom (pips)':   f"{avg_spread*10000:.4f}",
        'Sharpe (aprox)':       f"{sharpe:.3f}",
        'Max Drawdown':         f"${max_drawdown:.4f}",
        'σ calibrado':          f"{r.sigma_calibrated:.6f}/vela",
        'Retorno USD/MXN':      f"{price_ret:+.4f}%",
        'ψ utilizado':          f"{p.psi} MXN/USD",
    }


# ══════════════════════════════════════════════════════════════════
# 7. VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════

def plot_results(result: SimResult, stats: dict, fx_df: pd.DataFrame):
    """Dashboard con precios reales USD/MXN + quotes + métricas."""

    plt.style.use('dark_background')
    ACCENT  = '#00e5a0'
    RED     = '#ff4d6d'
    YELLOW  = '#fbbf24'
    PURPLE  = '#a78bfa'
    MUTED   = '#5a6478'
    BG      = '#0a0c0f'
    SURFACE = '#111318'

    fig = plt.figure(figsize=(17, 11), facecolor=BG)
    fig.suptitle(
        f'Market Making — Avellaneda-Stoikov  |  USD/MXN  '
        f'|  {result.times[0].strftime("%Y-%m-%d")} '
        f'→ {result.times[-1].strftime("%Y-%m-%d")}',
        fontsize=13, color='white', fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.48, wspace=0.35,
                           left=0.06, right=0.97,
                           top=0.93, bottom=0.06)

    r  = result
    p  = r.params
    ts = r.times

    # ── Panel 1: Precio USD/MXN + Quotes ─────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(SURFACE)

    # Rango High-Low como contexto
    ax1.fill_between(ts, fx_df['Low'].values, fx_df['High'].values,
                     alpha=0.10, color=YELLOW, label='Rango H-L')
    ax1.plot(ts, r.mid_prices, color=YELLOW, lw=1.5,
             label='Close (mid price)', zorder=3)
    ax1.plot(ts, r.reservation_prices, color=PURPLE, lw=1.0,
             ls='--', alpha=0.8, label='Precio Reserva r*(t)')
    ax1.fill_between(ts, r.bid_quotes, r.ask_quotes,
                     alpha=0.07, color=ACCENT)
    ax1.plot(ts, r.bid_quotes, color=ACCENT, lw=0.8, ls='--',
             alpha=0.7, label='Bid MM')
    ax1.plot(ts, r.ask_quotes, color=RED,    lw=0.8, ls='--',
             alpha=0.7, label='Ask MM')

    buys  = [t for t in r.trades if t.side == 'BUY']
    sells = [t for t in r.trades if t.side == 'SELL']
    if buys:
        ax1.scatter([ts[t.time_idx] for t in buys],
                    [t.price for t in buys],
                    color=ACCENT, marker='^', s=18, zorder=5,
                    alpha=0.7, label=f'Buys ({len(buys)})')
    if sells:
        ax1.scatter([ts[t.time_idx] for t in sells],
                    [t.price for t in sells],
                    color=RED, marker='v', s=18, zorder=5,
                    alpha=0.7, label=f'Sells ({len(sells)})')

    ax1.set_title('USD/MXN — Precio Real y Quotes del Market Maker',
                  color='white', fontsize=10, pad=6)
    ax1.set_ylabel('MXN / USD', color=MUTED, fontsize=8)
    ax1.legend(fontsize=6.5, loc='upper left', framealpha=0.3, ncol=3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20, ha='right')
    ax1.tick_params(colors=MUTED, labelsize=7)
    ax1.spines[:].set_color(MUTED); ax1.spines[:].set_alpha(0.3)
    ax1.grid(True, alpha=0.07, color='white')

    # ── Panel 2: Spread óptimo (en pips) ─────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.set_facecolor(SURFACE)
    spreads_pips = r.spreads * 10000
    ax2.plot(ts, spreads_pips, color=YELLOW, lw=1.3)
    ax2.fill_between(ts, 0, spreads_pips, alpha=0.15, color=YELLOW)
    ax2.set_title('Spread Óptimo δ*(t)  [pips = MXN × 10⁻⁴]',
                  color='white', fontsize=10, pad=6)
    ax2.set_ylabel('Pips', color=MUTED, fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=20, ha='right')
    ax2.tick_params(colors=MUTED, labelsize=7)
    ax2.spines[:].set_color(MUTED); ax2.spines[:].set_alpha(0.3)
    ax2.grid(True, alpha=0.07, color='white')

    mid_i = len(ts) // 2
    ax2.annotate('δ* ↓ al acercarse T\n(menos tiempo para rebalancear)',
                 xy=(ts[mid_i], spreads_pips[mid_i]),
                 xytext=(ts[max(0, mid_i - len(ts)//5)],
                         spreads_pips.max() * 0.85),
                 color=MUTED, fontsize=6.5,
                 arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.7))

    # ── Panel 3: Inventario (USD) ─────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.set_facecolor(SURFACE)
    inv_colors = np.where(r.inventories >= 0, ACCENT, RED)
    ax3.bar(ts, r.inventories,
            width=pd.Timedelta(seconds=p.dt * 0.85),
            color=inv_colors, alpha=0.7)
    ax3.axhline(0,  color=MUTED, lw=0.8, alpha=0.5)
    ax3.axhline( p.max_inventory, color=RED, lw=0.7, ls=':',
                alpha=0.4, label=f'Límite ±{p.max_inventory} USD')
    ax3.axhline(-p.max_inventory, color=RED, lw=0.7, ls=':', alpha=0.4)
    ax3.set_title('Inventario Neto q(t)  [USD]',
                  color='white', fontsize=10, pad=6)
    ax3.set_ylabel('USD', color=MUTED, fontsize=8)
    ax3.legend(fontsize=7, framealpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=20, ha='right')
    ax3.tick_params(colors=MUTED, labelsize=7)
    ax3.spines[:].set_color(MUTED); ax3.spines[:].set_alpha(0.3)
    ax3.grid(True, alpha=0.07, color='white')

    # ── Panel 4: PnL — modelo ψ·Turnover ────────────────────────
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.set_facecolor(SURFACE)

    # Bruto: retorno antes de descontar costo de turnover
    ax4.plot(ts, r.gross_returns, color=MUTED, lw=1.2, ls=':',
             alpha=0.8, label='Σ Rₜ bruto')

    # PnL neto: bruto − ψ·Turnover
    pnl_color = ACCENT if r.pnl_turnover[-1] >= 0 else RED
    ax4.plot(ts, r.pnl_turnover, color=pnl_color, lw=2.0,
             label='PnL  =  Σ Rₜ − ψ·Turnover')

    # Área que representa el costo ψ·Turnover
    ax4.fill_between(ts, r.pnl_turnover, r.gross_returns,
                     alpha=0.2, color=RED, label='Costo ψ·Turnover')

    ax4.axhline(0, color=MUTED, lw=0.8, ls='--', alpha=0.5)
    ax4.set_title('PnL  =  Σ Rₜ  −  ψ · Σ|Δq|  ',
                  color='white', fontsize=10, pad=6)
    ax4.set_ylabel('MXN', color=MUTED, fontsize=8)
    ax4.legend(fontsize=7, framealpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=20, ha='right')
    ax4.tick_params(colors=MUTED, labelsize=7)
    ax4.spines[:].set_color(MUTED); ax4.spines[:].set_alpha(0.3)
    ax4.grid(True, alpha=0.07, color='white')

    # ── Panel 5: Estadísticas ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[1:, 2])
    ax5.set_facecolor(SURFACE)
    ax5.axis('off')

    ax5.text(0.5, 0.99, 'Estadísticas', transform=ax5.transAxes,
             ha='center', va='top', color='white',
             fontsize=10, fontweight='bold')

    y_pos = 0.93
    for key, val in stats.items():
        vc = ('white' if not any(x in str(val) for x in ['$', '%', '/'])
              else (RED if '-' in str(val) else
                    (ACCENT if '$' in str(val) else MUTED)))

        ax5.text(0.04, y_pos, key + ':', transform=ax5.transAxes,
                 ha='left', va='top', color=MUTED, fontsize=7.5)
        ax5.text(0.97, y_pos, str(val), transform=ax5.transAxes,
                 ha='right', va='top', color=vc,
                 fontsize=7.5, fontweight='bold')
        y_pos -= 0.062

    # Parámetros del modelo
    y_pos -= 0.015
    ax5.text(0.5, y_pos, 'Parámetros', transform=ax5.transAxes,
             ha='center', va='top', color='white',
             fontsize=9, fontweight='bold')
    y_pos -= 0.06
    for line in [
        f'σ = {p.sigma:.2e} /s',
        f'γ = {p.gamma}   λ = {p.lam}   κ = {p.kappa}',
        f'T = {int(p.T)} velas  dt = {int(p.dt)} vela',
        f'q_max = ±{p.max_inventory} USD',
        f'ψ·Turnover: ψ = {p.psi} MXN/USD',

    ]:
        ax5.text(0.5, y_pos, line, transform=ax5.transAxes,
                 ha='center', va='top', color=MUTED, fontsize=7.0)
        y_pos -= 0.052

    plt.savefig('market_maker_usdmxn.png', dpi=150,
                bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.show()
    print("\n✓ Gráfica guardada como 'market_maker_usdmxn.png'")


# ══════════════════════════════════════════════════════════════════
# 8. ANÁLISIS DE SENSIBILIDAD
# ══════════════════════════════════════════════════════════════════

def sensitivity_analysis(prices: np.ndarray, timestamps: np.ndarray,
                         base_params: ModelParams, n_runs: int = 20):
    """
    Corre N simulaciones para cada γ sobre los mismos precios reales.
    Permite ver el trade-off entre PnL e inventario al variar la
    aversión al riesgo.
    """
    print("\n" + "═"*65)
    print("  ANÁLISIS DE SENSIBILIDAD — USD/MXN datos reales")
    print("═"*65)

    gammas = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    rows   = []

    for g in gammas:
        pnl_psis, max_invs = [], []
        p = ModelParams(
            S0=base_params.S0, sigma=base_params.sigma,
            gamma=g, lam=base_params.lam, kappa=base_params.kappa,
            T=base_params.T, dt=base_params.dt,
            max_inventory=base_params.max_inventory,
            fee_rate=base_params.fee_rate, fee_fixed=base_params.fee_fixed,
            psi=base_params.psi, seed=None,
        )
        sim = MarketMakingSimulator(p)
        for _ in range(n_runs):
            res = sim.simulate(prices, timestamps)
            pnl_psis.append(res.pnl_turnover[-1])
            max_invs.append(np.max(np.abs(res.inventories)))
        rows.append({
            'γ':                 g,
            'PnL ψ·Turn μ':      np.mean(pnl_psis),
            'PnL ψ·Turn σ':      np.std(pnl_psis),
            'Max |Inv| μ (USD)': np.mean(max_invs),
        })
        print(f"  γ={g:<5}  PnL_ψ={np.mean(pnl_psis):+.4f}  "
              f"MaxInv={np.mean(max_invs):.1f} USD")

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False,
                              float_format=lambda x: f"{x:.5f}"))
    return df


# ══════════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("╔" + "═"*62 + "╗")
    print("   MARKET MAKING SIMULATOR — USD/MXN  (yfinance)            ")
    print("╚" + "═"*62 + "╝\n")

    # ── 1. Descargar datos reales ─────────────────────────────────
    print("[ 1/4 ]  Descargando datos USD/MXN de Yahoo Finance ...")
    fx_df = download_fx(
        ticker   = "USDMXN=X",
        period   = "5d",       # últimos 5 días hábiles
        interval = "1m",       # velas de 1 minuto
    )
    prices     = fx_df['Close'].values.astype(float)
    timestamps = fx_df.index.to_pydatetime()

    # ── 2. Calibrar σ con datos reales ────────────────────────────
    print("\n[ 2/4 ]  Calibrando volatilidad ...")
    sigma_cal = calibrate_sigma(prices, dt_seconds=60.0)  # retorna σ por vela

    # ── 3. Configurar parámetros ──────────────────────────────────
    print("\n[ 3/4 ]  Configurando modelo ...")
    # ── Parámetros calibrados para USD/MXN con velas de 1m ────────
    #
    #  La clave del ajuste es que TODO está en unidades de VELAS (dt=1):
    #
    #  σ_vela  ≈ 0.0002   (volatilidad por vela de 1m, calibrada arriba)
    #  γ = 0.1            → spread ≈ γ·σ²·T ≈ 0.1×(0.0002)²×7000 ≈ 0.000028 MXN
    #                        = 0.28 pips  (muy realista para USD/MXN)
    #  κ = 5000           → un quote a 0.0002 MXN del mid tiene:
    #                        intensidad × exp(-5000×0.0002) = ×37% (razonable)
    #  λ = 0.3            → ~0.3 órdenes por vela = ~18 trades/hora
    #
    params = ModelParams(
        S0            = float(prices[0]),
        sigma         = sigma_cal,   # σ por VELA, calibrado con datos reales
        gamma         = 0.1,         # aversión al riesgo (bajo → spreads angostos)
        lam           = 0.3,         # λ órdenes por VELA (~18 trades/hora)
        kappa         = 5000.0,      # κ ≈ 1/spread_típico_MXN (spread~0.0002)
        T             = float(len(prices)),  # horizonte = total de velas
        dt            = 1.0,         # 1 paso = 1 vela (escala consistente)
        max_inventory = 100,         # máx 100 USD de inventario neto
        fee_rate      = 0.0002,      # 2 bps por trade (broker retail FX)
        fee_fixed     = 0.0,
        psi           = 0.0005,      # half-spread ≈ 0.5 pip USD/MXN
        seed          = 42,
    )

    print(f"  S0             = {params.S0:.4f} MXN/USD")
    print(f"  σ por vela     = {params.sigma:.6f}  (calibrado)")
    print(f"  γ              = {params.gamma}  →  spread ≈ {params.gamma * params.sigma**2 * len(prices) * 10000:.2f} pips")
    print(f"  λ / κ          = {params.lam} / {params.kappa}")
    print(f"  Horizonte T    = {int(params.T)} velas  ({params.T/390:.1f} días)")
    print(f"  fee_rate       = {params.fee_rate*100:.3f}%  (2 bps)")
    print(f"  ψ (psi)        = {params.psi} MXN/USD")

    # ── 4. Simular ────────────────────────────────────────────────
    print("\n[ 4/4 ]  Ejecutando simulación sobre precios reales ...")
    sim    = MarketMakingSimulator(params)
    result = sim.simulate(prices, timestamps)

    n_trades = len(result.trades)
    print(f"  ✓ Simulación completada  |  {n_trades} trades ejecutados")

    # ── Estadísticas ──────────────────────────────────────────────
    stats = compute_stats(result)

    print("\n" + "═"*48)
    print("  RESULTADOS")
    print("═"*48)
    for k, v in stats.items():
        if v == '':
            print(f"\n  {k}")
        else:
            print(f"  {k:<28} {v}")

    # ── Dashboard ─────────────────────────────────────────────────
    print("\n▶  Generando dashboard ...")
    plot_results(result, stats, fx_df)

    # ── Sensibilidad (opcional) ───────────────────────────────────
    ans = input("\n¿Correr análisis de sensibilidad sobre γ? (s/n): ").strip().lower()
    if ans == 's':
        n = int(input("  Número de runs por γ (recomendado 20-50): ") or "20")
        sensitivity_analysis(prices, timestamps, params, n_runs=n)

    print("\n✓ Todo listo.")


if __name__ == "__main__":
    main()
