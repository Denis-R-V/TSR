def parse_thr(text: str) -> float:
    t = text.strip().replace(",", ".")
    v = float(t)
    if not (0.0 <= v <= 1.0):
        raise ValueError("thr out of range")
    return v
