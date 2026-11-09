"""Whitelisted dimensional arithmetic over cited facts; no model-authored formulas."""
import math

from .schemas import Quantity

UNITS = {
    "byte": ("byte", 1), "B": ("byte", 1), "Bytes": ("byte", 1),
    "KB": ("byte", 1e3), "MB": ("byte", 1e6), "GB": ("byte", 1e9),
    "TB": ("byte", 1e12), "KiB": ("byte", 1024), "MiB": ("byte", 1024**2),
    "GiB": ("byte", 1024**3),
    "byte/s": ("byte/s", 1), "B/s": ("byte/s", 1), "Bytes/sec": ("byte/s", 1),
    "GB/s": ("byte/s", 1e9), "TB/s": ("byte/s", 1e12),
    "GB/sec": ("byte/s", 1e9), "TB/sec": ("byte/s", 1e12),
    "GB/s/device": ("byte/s", 1e9),
    "FLOP/s": ("FLOP/s", 1), "GFLOP/s": ("FLOP/s", 1e9),
    "TFLOP/s": ("FLOP/s", 1e12), "TFLOPS": ("FLOP/s", 1e12),
}
CLOCK_UNITS = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}


def calculate(facts, fact_ids, derivation, target):
    """Return the canonical value and the exact conversion scale, or reject."""
    if not fact_ids or any(fid not in facts for fid in fact_ids):
        raise ValueError("Quantity references an absent source fact")
    selected = [facts[fid] for fid in fact_ids]
    if len({f.variant for f in selected}) != 1:
        raise ValueError("Arithmetic cannot combine hardware variants")
    fact = selected[0]
    if derivation == "unit_conversion":
        if len(selected) != 1:
            raise ValueError("Unit conversion requires one source fact")
        conversion = UNITS.get(fact.unit)
        if conversion is None or conversion[0] != target:
            raise ValueError(f"Unsupported unit conversion: {fact.unit} to {target}")
        if fact.aggregation != "device":
            raise ValueError("Device model needs device-level evidence or explicit aggregation")
        value, scale = fact.value * conversion[1], conversion[1]
    elif derivation == "aggregate_capacity":
        if len(selected) != 2 or target != "byte":
            raise ValueError("Capacity aggregation requires a per-component capacity and a count")
        count = selected[1]
        base, slash, component = fact.unit.partition("/")
        if (fact.aggregation != "component" or not slash or component != "SM"
                or count.unit not in {"SM", "SMs"} or count.parameter != "sm_count"
                or count.value != int(count.value)):
            raise ValueError("Only explicit per-SM capacity times the product's SM count is supported")
        conversion = UNITS.get(base)
        if conversion is None or conversion[0] != "byte":
            raise ValueError("Unsupported per-SM capacity unit")
        scale = conversion[1]
        value = fact.value * count.value * scale
    elif derivation == "per_clock_rate":
        if len(selected) != 2 or target != "byte/s":
            raise ValueError("Per-clock conversion requires bandwidth and a clock")
        clock = selected[1]
        if (fact.unit not in {"Bytes/clk", "bytes/clk", "B/clk", "byte/clock"}
                or fact.aggregation != "device" or clock.unit not in CLOCK_UNITS
                or clock.parameter not in {"clock_frequency", "gpu_clock", "boost_clock"}
                or clock.aggregation != "device"):
            raise ValueError("Per-clock bandwidth needs a device-scoped, documented GPU clock")
        scale = CLOCK_UNITS[clock.unit]
        value = fact.value * clock.value * scale
    else:
        raise ValueError("Unsupported arithmetic operation")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Invalid derived quantity")
    return value, scale


def derive(reference, facts, target):
    if reference.count_fact_id and reference.clock_fact_id:
        raise ValueError("A quantity cannot combine count and clock conversions")
    if reference.assumption_id:
        raise ValueError("Assumptions must be resolved separately")
    if reference.fact_id is None:
        raise ValueError("No documented value or explicit assumption was supplied")
    if (reference.fact_id in facts and target == "byte/s"
            and facts[reference.fact_id].unit in {"Bytes/clk", "bytes/clk", "B/clk", "byte/clock"}
            and not reference.clock_fact_id):
        raise ValueError("Documented bytes/clock rate needs a cited clock for conversion to byte/s")
    ids = [reference.fact_id] if reference.fact_id else []
    operation = "unit_conversion"
    if reference.count_fact_id:
        ids.append(reference.count_fact_id)
        operation = "aggregate_capacity"
    if reference.clock_fact_id:
        ids.append(reference.clock_fact_id)
        operation = "per_clock_rate"
    value, scale = calculate(facts, ids, operation, target)
    return Quantity(value=value, unit=target, basis="derived" if operation != "unit_conversion"
                    or scale != 1 else "documented", fact_ids=ids, scale=scale,
                    derivation=operation, rationale=f"Verified {operation} from " + ", ".join(ids))
