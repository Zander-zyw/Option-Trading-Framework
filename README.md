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

### 2. Straddle Strategy (Planned)
[Strategy description to be added]

### 3. Delta Neutral Strategy (Planned)
[Strategy description to be added]

## Requirements

- Python 3.7+
- Required packages:
  - aiohttp>=3.8.0
  - websockets>=10.0
  - cryptography>=3.4.7

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Option-Trading-Framework.git
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

### Running Cover Call Strategy

1. Start the strategy:
```bash
python Strategies/Cover_Call.py
```

2. Monitor the strategy:
- Logs are written to the configured logging destination
- State is saved in `state/cover_call_state_{symbol}.json`

3. Stop the strategy:
- Press Ctrl+C for graceful shutdown
- Or send SIGTERM signal:
```bash
pkill -f "python Strategies/Cover_Call.py"
```

## State Management

The framework automatically:
- Saves current positions to `state/cover_call_state_{symbol}.json`
- Loads previous state on restart
- Maintains position tracking during runtime

## Logging

Comprehensive logging is implemented for:
- Strategy execution
- Position management
- Error handling
- State changes
- Order execution

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Currently None License]

## Disclaimer

This framework is for educational purposes only. Use at your own risk. Past performance is not indicative of future results.