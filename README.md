Simulador de estrategia de market making óptima implementando el modelo 
de Avellaneda & Stoikov (2008) sobre datos reales de USD/MXN obtenidos 
via yfinance.

El market maker coloca quotes bid/ask óptimos en cada minuto, ajustando 
sus precios según el inventario acumulado y el tiempo restante del horizonte 
de trading. El costo de transacción se modela con el enfoque de ψ·Turnover 
estándar en la industria.

Características:
- Descarga automática de precios reales FX (Yahoo Finance)
- Calibración de volatilidad σ con datos históricos
- Modelo de precio de reserva y spread óptimo (A-S 2008)
- Simulación de llegada de órdenes via proceso de Poisson
- Costo de transacción: modelo ψ·Turnover
- Análisis de sensibilidad sobre aversión al riesgo γ
- Dashboard completo con matplotlib

Stack: Python · NumPy · Pandas · Matplotlib · yfinance

Instalación:
    pip install numpy pandas matplotlib yfinance
    python market_maker.py
