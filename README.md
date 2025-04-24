# Option Trading Framework

A robust framework for automated option trading strategies, currently supporting Deribit exchange.

## Features

- Automated option trading strategies
- Position management with dynamic leverage
- Real-time IV monitoring
- Graceful shutdown and state persistence
- Comprehensive logging
- Risk management with stop-loss mechanisms

## Implemented Strategies

### 1. Cover Call Strategy
A strategy that sells call options when IV is high, with dynamic position sizing and stop-loss management.

#### Features
- Dynamic position sizing based on IV levels:
  - IV >= 55: 0.5x leverage (半仓)
  - IV >= 65: 1.0x leverage (满仓)
  - IV >= 75: 1.5x leverage (1.5倍杠杆)
- Stop-loss mechanism based on settlement price (default: 4x)
- Call level setting (default: 1.2x)
- Position monitoring and automatic closing
- State persistence and recovery
- Graceful shutdown handling

### 2. Straddle Strategy
A market-neutral options strategy that involves simultaneously buying a put and a call with the same strike price and expiration date.

#### Features
- Dynamic entry based on IV levels and term structure
- Real-time volatility surface monitoring
- Position sizing based on account risk parameters
- Automatic strike selection near ATM (At-The-Money)
- Stop-loss based on total position equity
- Profit taking at predefined profit targets
- State persistence and recovery
- Graceful shutdown handling

### 3. Delta Neutral Strategy
A strategy that maintains a delta-neutral position by dynamically hedging options positions with the underlying asset.

#### Features
- Continuous delta calculation and monitoring
- Dynamic hedge ratio adjustment
- Automated futures hedging
- Position correlation monitoring
- Risk metrics tracking (Vega, Gamma, Theta)
- State persistence and recovery
- Graceful shutdown handling


## Requirements

- Python 3.7+
- Required packages:
  - aiohttp>=3.8.0
  - websockets>=10.0
  - cryptography>=3.4.7
  - python-okx>=0.3.5

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Zander-zyw/Option-Trading-Framework.git
cd Option-Trading-Framework
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### Cover Call Strategy Configuration
```python
symbol = "BTC"  # Trading symbol
position_thresholds = {
    55: 0.5,  # 半仓
    65: 1.0,  # 满仓
    75: 1.5   # 1.5倍杠杆
}
stop_loss_multiplier = 4.0  # Stop loss multiplier
call_level = 1.2  # Call level
```

## Usage

### Running Strategies

1. Cover Call Strategy:
```bash
python Strategies/Deribit/Cover_Call.py
```

2. Straddle Strategy:
```bash
python Strategies/Deribit/Straddle.py
```
```bash
python Strategies/OKX/Straddle.py
```

3. Delta Neutral Strategy:
```bash
python Strategies/Deribit/Delta_Neutral.py
```

Monitor the strategies:
- Logs are written to the configured logging destination
- States are saved in `state/{strategy_name}_state_{symbol}.json`

Stop a strategy:
- Press Ctrl+C for graceful shutdown
- Or send SIGTERM signal:
```bash
pkill -f "python Strategies/{exchange_name}/{strategy_name}.py"
```

## State Management

The framework automatically:
- Saves current positions to `state/{strategy_name}_state_{symbol}.json`
- Loads previous state on restart
- Maintains position tracking during runtime

## Logging

Comprehensive logging is implemented for:
- Strategy execution
- Position management
- Error handling
- State changes
- Order execution
- Risk metrics
- Hedging activities
- Rebalancing operations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Currently None License]

## Disclaimer

This framework is for educational purposes only. Use at your own risk. Past performance is not indicative of future results. Options trading involves substantial risk and is not suitable for all investors.