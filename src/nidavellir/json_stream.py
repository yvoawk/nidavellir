"""Atomic JSON output with lazy arrays and bounded record buffering."""
import json
import time


class JsonArray:
    """Mark a lazily generated JSON array without materializing its values."""

    def __init__(self, values):
        self.values = values


def write_json(path, data, *, deadline=None):
    """Encode individual records in C; retain streaming for nested lazy arrays.

    At most 128 encoded records or 256K characters (plus one record) are buffered.
    No whole lazy array is copied, and failed writes never replace the target.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")

    def check():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Preparation exceeded its wall-clock budget")

    def encode(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def emit(stream, value):
        check()
        if isinstance(value, dict):
            stream.write("{")
            for position, (key, item) in enumerate(value.items()):
                if position:
                    stream.write(",")
                stream.write(encode(key) + ":")
                emit(stream, item)
            stream.write("}")
        elif isinstance(value, (list, tuple, JsonArray)):
            stream.write("[")
            values = value.values if isinstance(value, JsonArray) else value
            pending, size = [], 0
            for position, item in enumerate(values):
                check()
                try:
                    record = ("," if position else "") + encode(item)
                except TypeError:
                    # json.dumps cannot consume JsonArray, so retrying the
                    # nested structure cannot lose or repeat generator items.
                    stream.write("".join(pending))
                    pending, size = [], 0
                    if position:
                        stream.write(",")
                    emit(stream, item)
                    continue
                pending.append(record)
                size += len(record)
                if size >= 256 * 1024 or len(pending) >= 128:
                    stream.write("".join(pending))
                    pending, size = [], 0
            stream.write("".join(pending))
            stream.write("]")
        else:
            stream.write(encode(value))

    try:
        with temporary.open("w", buffering=1024 * 1024) as stream:
            emit(stream, data)
            stream.write("\n")
            check()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
