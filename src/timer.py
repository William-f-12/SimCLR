class EMAMeter:
    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self.ema = None

    def update(self, x: float):
        if self.ema is None:
            self.ema = x
        else:
            self.ema = self.beta * self.ema + (1 - self.beta) * x

    @property
    def avg(self):
        return self.ema if self.ema is not None else 0.0

def format_time(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0: return f"{h}h{m:02d}m{s:02d}s"
    if m > 0: return f"{m}m{s:02d}s"
    return f"{s}s"