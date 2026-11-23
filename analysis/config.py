"""Fixed analysis window, so the published numbers reproduce exactly even though the
collector branches may hold more data than was analysed."""
import pandas as pd

WINDOW_START_UTC = "2026-11-23T17:06:00Z"
WINDOW_END_UTC = "2026-11-23T21:00:00Z"
START_MS = int(pd.Timestamp(WINDOW_START_UTC).value // 10**6)
END_MS = int(pd.Timestamp(WINDOW_END_UTC).value // 10**6)
TIME_COL = {"quotes": "t_recv", "ladder": "t_recv", "book": "t_recv", "trades": "t", "fills": "time",
            "mids": "t_recv", "ctx": "t_recv", "pos": "t_recv", "horders": "t_recv"}
