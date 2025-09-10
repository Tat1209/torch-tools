import polars as pl

def split_pm(df: pl.DataFrame) -> pl.DataFrame:
    metrics_columns = [name for name, dtype in df.schema.items() if isinstance(dtype, pl.List)]
    df_params = df.select(pl.exclude(metrics_columns))
    df_metrics = df.select(metrics_columns)
    df_metrics = df_metrics.explode(pl.all())
    
    return df_params, df_metrics

def get_stats(df: pl.DataFrame) -> pl.DataFrame:
    nested_columns = [name for name, dtype in zip(df.columns, df.dtypes) if isinstance(dtype, pl.List)]
    df = df.with_columns([pl.col(name).list.last() for name in nested_columns])
    # df_base = df_base.with_columns([pl.col(name).list.last().alias(f"{name}") for name in nested_columns])
    
    return df

def get_stat(df: pl.DataFrame) -> pl.DataFrame:
    df_params, df_metrics = df.pipe(split_pm)
    df = df_params.join(df_metrics, how="cross")
    
    return df

def add_iter_step(df: pl.DataFrame,step_col="step", iter_ndata_col="iter_ndata", new_col_name="iter_step") -> pl.DataFrame:
    df = df.with_columns(
        pl.struct([iter_ndata_col, step_col]).map_elements(
            lambda s: [
                [
                    # sum(sublist)が0になる場合のゼロ除算を避ける
                    (
                        sum(sublist[:j+1]) / sum(sublist)
                        if sum(sublist) != 0
                        else 0.0
                    ) + s[step_col][i] - 1
                    for j in range(len(sublist))
                ]
                for i, sublist in enumerate(s[iter_ndata_col])
            ],
            # map_elementsではreturn_dtypeの指定が推奨されます
            return_dtype=pl.List(pl.List(pl.Float64))
        ).alias(new_col_name)
    )
    return df

def unnest_iter(df: pl.DataFrame, iter_epoch_col="iter_step") -> pl.DataFrame:
    """
    データフレーム内で指定されたカラムと形状が一致する、
    2段階ネストされたList[List[Float64]]型のカラム（基準カラム自身も含む）を1段階アンネストします。
    """
    if iter_epoch_col not in df.columns or not isinstance(df.schema[iter_epoch_col], pl.List):
        return df

    cols_to_unnest = []
    
    for col_name in df.columns:
        
        col_dtype = df.schema[col_name]
        
        is_nested_list_f64 = (
            isinstance(col_dtype, pl.List) and
            isinstance(col_dtype.inner, pl.List) and
            col_dtype.inner.inner == pl.Float64
        )
        
        if is_nested_list_f64:
            try:
                # 2. リストの長さが基準カラムと全行で一致するかチェック
                #    (自分自身との比較もここに含まれる)
                is_len_equal = df.select(
                    (pl.col(col_name).list.len() == pl.col(iter_epoch_col).list.len()).all()
                ).item()
                
                if is_len_equal:
                    cols_to_unnest.append(col_name)

            except pl.InvalidOperationError:
                continue

    if cols_to_unnest:
        df = df.with_columns(
            pl.col(cols_to_unnest).list.eval(pl.element().explode())
        )
        
    return df
    


class Config:
    def __init__(self, **overrides):
        self.defaults = {
            "tbl_rows": 10,
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