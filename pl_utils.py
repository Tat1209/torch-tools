import polars as pl

def split_pm(df_row: pl.DataFrame) -> pl.DataFrame:
    metrics_columns = [name for name, dtype in df_row.schema.items() if isinstance(dtype, pl.List)]
    df_params = df_row.select(pl.exclude(metrics_columns))
    df_metrics = df_row.select(metrics_columns)
    df_metrics = df_metrics.explode(pl.all())
    
    return df_params, df_metrics

def get_stats(df: pl.DataFrame) -> pl.DataFrame:
    nested_columns = [name for name, dtype in zip(df.columns, df.dtypes) if isinstance(dtype, pl.List)]
    df = df.with_columns([pl.col(name).list.last() for name in nested_columns])
    # df_base = df_base.with_columns([pl.col(name).list.last().alias(f"{name}") for name in nested_columns])
    
    return df

def get_stat(df_row: pl.DataFrame) -> pl.DataFrame:
    df_params, df_metrics = df_row.pipe(split_pm)
    df = df_params.join(df_metrics, how="cross")
    
    return df

class Config:
    def __init__(self, **overrides):
        self.defaults = {
            "tbl_cols": -1,
            "fmt_str_lengths": 24,
            "tbl_width_chars": -1,
            "tbl_cell_numeric_alignment": "RIGHT",
        }
        self.options = {**self.defaults, **overrides}
        self.cfg = pl.Config(**self.options)

        self.cfg.__enter__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cfg.__exit__(exc_type, exc_val, exc_tb)