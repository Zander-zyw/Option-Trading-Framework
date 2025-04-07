# Option Trading Framework

A robust framework for automated option trading strategies, currently supporting Deribit exchange.

## Features

- Automated option trading strategies
- Position management with dynamic leverage
- Real-time IV monitoring
- Graceful shutdown and state persistence
- Comprehensive logging
- Risk management with stop-loss mechanisms

## Current Strategy: Cover Call

The framework currently implements a Cover Call strategy with the following features:

### Position Management
- Dynamic position sizing based on IV levels:
  - IV >= 55: 0.5x leverage (半仓)
  - IV >= 65: 1.0x leverage (满仓)
  - IV >= 75: 1.5x leverage (1.5倍杠杆)

### Risk Management
- Stop-loss mechanism based on settlement price
- Configurable stop-loss multiplier (default: 4x)
- Position monitoring and automatic closing

### State Management
- Automatic state saving on shutdown
- State restoration on restart
- Position tracking and management

## Requirements

- Python 3.7+
- Required packages (to be added to requirements.txt)

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

The strategy can be configured through the following parameters:

```python
symbol = "BTC"  # Trading symbol
position_thresholds = {
    55: 0.5,  # 半仓
    65: 1.0,  # 满仓
    75: 1.5   # 1.5倍杠杆
}
stop_loss_multiplier = 4.0  # Stop loss multiplier
```

## Usage

1. Start the strategy:
```bash
python Strategies/Cover_Call.py
```

2. Monitor the strategy:
- Logs are written to the configured logging destination
- State is saved in `State/cover_call_state_{symbol}.json`

3. Stop the strategy:
- Press Ctrl+C for graceful shutdown
- Or send SIGTERM signal:
```bash
pkill -f "python Strategies/Cover_Call.py"
```

## State Management

The framework automatically:
- Saves current positions to `State/cover_call_state_{symbol}.json`
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