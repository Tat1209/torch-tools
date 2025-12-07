import glob
import shutil
from pathlib import Path

import numpy as np
import polars as pl

from utils import interval


class RunManager:
    params_pq = "_params.parquet"
    metrics_pq = "_metrics.parquet"
    results_pq = "_results.parquet"

    params_csv = "_params.csv"
    metrics_csv = "_metrics.csv"
    results_csv = "_results.csv"

    def __init__(self, exc_path, exp_tpl="exp_tpl"):
        """
        ex.)
            run = RunManager(exc_path=__file__, exp_name="exp_nyancat")
        """
        self.run_path = exc_path
        if not exc_path:
            exc_path = Path().cwd()
        
        try:
            path_exp_def = Path(__file__).resolve().parent / "exp_tpls" / exp_tpl
            if path_exp_def.is_dir():
                for item in path_exp_def.iterdir():
                    dest = self.exp_path / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    elif item.is_file():
                        shutil.copy2(item, dest)
        except Exception:
            pass

        self.df_params = None
        self.df_metrics = None

    def fpath(self, fname):
        return self.run_path / Path(fname)
        
    def log_text(self, fname, text):
        with open(self.fpath(fname), "w") as fh:
            fh.write(text)

    def log_torch_save(self, object, fname):
        import torch
        torch.save(object, self.fpath(fname))
        
    def log_param(self, name, value):
        self.log_params({name: value})

    def log_params(self, stored_dict):
        processed_dict = {}
        for k, v in stored_dict.items():
            try:
            # paramがlistのとき，explodeの処理をかけたくないのにかけられる．Arrayに変換して区別できるように
                if isinstance(v, list):
                    v = np.array(v)
                processed_dict[k] = [v]
            except Exception as e:
                raise ValueError(f"キー '{k}' の値を NumPy 配列に変換できませんでした。すべての要素が同じ型である必要があります。") from e

        df_new = pl.DataFrame(processed_dict)
        if self.df_params is None:
            self.df_params = df_new
        else:
            self.df_params = self.df_params.hstack(df_new)

    def log_metric(self, name, value, step):
        self.log_metrics({name: value}, step=step)

    def log_metrics(self, stored_dict, step):
        if self.df_metrics is None:
            self.df_metrics = pl.DataFrame({"step": [step], **{k: [v] for k, v in stored_dict.items()}})
            return
        
        step_exists = self.df_metrics["step"].eq(step).any() # step の存在を確認

        if step_exists:
            for name, value in stored_dict.items():
                self.df_metrics = self._df_set_elem(self.df_metrics, name, "step", step, value)
        else:
            df_tmp = pl.DataFrame({"step": [step], **{k: [v] for k, v in stored_dict.items()}})
            self.df_metrics = pl.concat([self.df_metrics, df_tmp], how="diagonal_relaxed")

    def _df_set_elem(self, df, column, index_column_name, index, value):
        # index_column_name で指定した列の値が index の行の column に value を設定
        if value is None:
            return df
        
        if column in df.columns:
            return df.with_columns(
                pl.when(df[index_column_name] == index)
                .then(pl.lit(value))
                .otherwise(pl.col(column))
                .alias(column)
            )
        else:
            return df.with_columns(
                pl.when(df[index_column_name] == index)
                .then(pl.lit(value))
                .otherwise(pl.lit(None))
                .alias(column)
            )
    
    def _inherit_stats(self):
        if self.fpath(self.params_pq).exists():
            self.df_params = pl.read_parquet(self.fpath(self.params_pq))
        if self.fpath(self.metrics_pq).exists():
            self.df_metrics = pl.read_parquet(self.fpath(self.metrics_pq))
            
    def ref_stats(self, step=None, itv=None, last_step=None):
        if interval(step=step, itv=itv, last_step=last_step):
            if self.df_params is not None:
                self.df_params.write_parquet(self.fpath(self.params_pq))
                df_nonest = self.df_params.pipe(self._resolve_nested)
                df_nonest.write_csv(self.fpath(self.params_csv))
            if self.df_metrics is not None:
                self.df_metrics.write_parquet(self.fpath(self.metrics_pq))
                df_nonest = self.df_metrics.pipe(self._resolve_nested)
                df_nonest.write_csv(self.fpath(self.metrics_csv))

    @classmethod
    def _collect_runs(cls, target_path, prefix="run_", sort_subdir=True):
        search_root = Path(target_path).resolve()
        
        run_containers = [p for p in search_root.rglob(f"{prefix}*") if p.is_dir()]
        run_containers.sort(key=lambda p: p.name)
        
        sorted_leaf_dirs = []

        for container in run_containers:
            found_params = list(container.rglob(cls.params_pq))
            
            def _key(p):
                d = p.parent
                if d == container:
                    return -1
                if sort_subdir and d.name.isdigit():
                    return int(d.name)
                return str(d.name)

            found_params.sort(key=_key)
            sorted_leaf_dirs.extend([p.parent for p in found_params])
            
        return sorted_leaf_dirs

    @staticmethod
    def _resolve_nested(df):
        nested_columns = [name for name, dtype in zip(df.columns, df.dtypes) if dtype.is_nested()]
        for name in nested_columns:
            try:
                df = df.with_columns([(pl.lit("last: ") + pl.col(name).list.last().cast(pl.Utf8)).alias(name)])
            except (pl.exceptions.InvalidOperationError, pl.exceptions.SchemaError):
                df = df.with_columns([pl.lit("(nested_data)").alias(name)])
        return df

    @classmethod
    def ref_results(cls, target_path, step=None, itv=None, last_step=None, refresh=True, sort_subdir=False):
        if interval(step=step, itv=itv, last_step=last_step):
            # ログを直下に持つディレクトリを収集し，ルールを用いてsort
            run_dirs = cls._collect_runs(target_path, prefix="run_", sort_subdir=sort_subdir)
            
            stats_l = []
            for run_dir in run_dirs:
                params_path = run_dir / cls.params_pq
                metrics_path = run_dir / cls.metrics_pq
                
                df_run = pl.DataFrame({"run_path": str(run_dir.resolve())}, schema=pl.Schema({"run_path": pl.Utf8}))
                
                if params_path.exists():
                    df_params = pl.read_parquet(params_path)
                    df_run = df_run.hstack(df_params)
                
                if metrics_path.exists():
                    df_metrics = pl.read_parquet(metrics_path)
                    df_run = df_run.hstack(df_metrics.select(pl.all().implode()))
                
                stats_l.append(df_run)
            
            if not stats_l:
                return None
            
            df = pl.concat(stats_l, how="diagonal_relaxed")

            if refresh:
                df.write_parquet(Path(target_path) / cls.results_pq)
                df_nonest = df.pipe(cls._resolve_nested)
                df_nonest.write_csv(Path(target_path) / cls.results_csv)
                return df
            else:
                p = Path(target_path) / cls.results_pq
                if p.exists():
                    return pl.read_parquet(p)
                return df
                 

# これにはrun_idつくるのが妥当? めんどいけど
class RunsManager:
    def __init__(self, runs):
        self.runs = runs
    
    def __getattr__(self, attr):
        def wrapper(*args, **kwargs):
            n = len(self.runs)
            new_args = [[] for _ in range(n)]
            new_kwargs = [{} for _ in range(n)]

            for arg in args:
                if isinstance(arg, list) and len(arg) == n:
                    for i in range(n):
                        new_args[i].append(arg[i])

                elif isinstance(arg, dict):
                    per_dicts = [ {} for _ in range(n) ]
                    for k, v in arg.items():
                        pairable = isinstance(v, list) and len(v) == n
                        for i in range(n):
                            per_dicts[i][k] = v[i] if pairable else v
                    for i in range(n):
                        new_args[i].append(per_dicts[i])

                else:
                    for i in range(n):
                        new_args[i].append(arg)

            for kw, val in kwargs.items():
                if isinstance(val, list) and len(val) == n:
                    for i in range(n):
                        new_kwargs[i][kw] = val[i]

                elif isinstance(val, dict):
                    per_dicts = [{} for _ in range(n)]
                    for k, v in val.items():
                        pairable = isinstance(v, list) and len(v) == n
                        for i in range(n):
                            per_dicts[i][k] = v[i] if pairable else v
                    for i in range(n):
                        new_kwargs[i][kw] = per_dicts[i]

                else:
                    for i in range(n):
                        new_kwargs[i][kw] = val

            results = []
            for run, a, kw in zip(self.runs, new_args, new_kwargs):
                results.append(getattr(run, attr)(*a, **kw))
            return results
        return wrapper

    def __getitem__(self, idx):
        return self.runs[idx]

def cat_results(exp_paths, refresh=False):
    """
    指定されたパスパターンに一致するすべての結果ファイルを連結して一つのDataFrameにします。

    Args:
        exp_paths (list of str or Path): ファイルパス/パターンのリスト。ワイルドカード ('*', '**') を含むことが可能。
        refresh (bool, optional): RunViewerのキャッシュをリフレッシュするかどうか。デフォルトは False。

    Returns:
        polars.DataFrame: 連結された結果のDataFrame。
    """
    file_paths = set()
    for pattern in exp_paths:
        file_paths.update(glob.glob(str(pattern), recursive=True))

    dfs = []
    for file_path_str in sorted(list(file_paths)):
        exp_path = Path(file_path_str)
        
        if exp_path.is_dir():
            exp_name = exp_path.name
            df = RunManager.ref_results(target_path=exp_path, refresh=refresh)
            df = df.with_columns(pl.lit(exp_name).alias("exp_name"))
            df = df.select(["exp_name", *df.columns[:-1]])
            dfs.append(df)
            
    dfs_cat = pl.concat(dfs, how="diagonal_relaxed")
    
    return dfs_cat

