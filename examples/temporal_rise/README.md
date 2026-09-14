# Temporal sensor-rise example

This synthetic task learns `rise(Device, Time)` from timestamped sensor
readings. The BK defines a 1.5x increase relation and strict temporal order. The
bias permits a four-literal rule joining the current reading to an earlier
reading through hidden value and time slots.

From the repository root:

```bash
python -m nidavellir \
  --background examples/temporal_rise/background.lp \
  --examples examples/temporal_rise/examples.lp \
  --bias examples/temporal_rise/bias.lp \
  --output /tmp/nidavellir-temporal
```

The expected complete result is one rule of cost five. The example is designed
to test correlated existential joins rather than independent literal matches.
