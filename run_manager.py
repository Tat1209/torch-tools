import glob
import shutil
from pathlib import Path

import numpy as np
import polars as pl

from utils import interval
from utils import format_time


class RunManager:
    params_pq = "_params.parquet"
    metrics_pq = "_metrics.parquet"
    results_pq = "_results.parquet"

    params_csv = "_params.csv"
    metrics_csv = "_metrics.csv"
    results_csv = "_results.csv"

    def __init__(self, exc_path, exp_name="exp_default", exp_tpl="exp_tpl"):
        """
        ex.)
            run = RunManager(exc_path=__file__, exp_name="exp_nyancat")
        """
        self.exc_path = exc_path
        self.exp_path = Path(exc_path).resolve().parent / exp_name
        self.runs_path = self.exp_path / "runs"
        run_id = f"run_{format_time(style=2)}"
        run_path = self.runs_path / run_id

        self.run_id = run_id
        self.run_path = run_path
        self.df_params = None
        self.df_metrics = None
        
        if not self.exp_path.exists():
            self.exp_path.mkdir(parents=True, exist_ok=False)
            path_exp_def = Path(__file__).resolve().parent / "exp_tpls" / exp_tpl
            if path_exp_def.is_dir():
                for item in path_exp_def.iterdir():
                    dest = self.exp_path / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    elif item.is_file():
                        shutil.copy2(item, dest)
        self.runs_path.mkdir(parents=True, exist_ok=True)

        # if run_id is None:
        #     run_id = self._get_run_id(self.runs_path)
        # else:
        #     self._inherit_stats()

        self.run_path.mkdir(parents=True, exist_ok=True)


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

    def _resolve_nested(self, df):
        nested_columns = [name for name, dtype in zip(df.columns, df.dtypes) if dtype.is_nested()]
        for name in nested_columns:
            try:
                df = df.with_columns([(pl.lit("last: ") + pl.col(name).list.last().cast(pl.Utf8)).alias(name)])
            except (pl.exceptions.InvalidOperationError, pl.exceptions.SchemaError):
                df = df.with_columns([pl.lit("(nested_data)").alias(name)])
        return df
    
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

    def fetch_files(self, fname):
        dir_names = list(self.runs_path.iterdir())
        run_ids = [dir_name.name for dir_name in dir_names]
        stats_paths = [dir_name / Path(fname) for dir_name in dir_names]
        
        return run_ids, stats_paths

    def ref_results(self, step=None, itv=None, last_step=None, fname=None, refresh=True):
        if fname is None:
            fname = self.results_pq
        if interval(step=step, itv=itv, last_step=last_step):
            run_ids, params_paths = self.fetch_files(self.params_pq)
            run_ids, metrics_paths = self.fetch_files(self.metrics_pq)

            stats_l = []
            for run_id, params_path, metrics_path in zip(run_ids, params_paths, metrics_paths):
                df_run = pl.DataFrame({"run_id": run_id})
                if params_path.exists() and params_path.stat().st_size > 12:
                    try:
                        df_params = pl.read_parquet(params_path)
                    except Exception as e:
                        print(f"Failed to read params parquet for run_id {run_id}")
                        continue
                    df_run = df_run.hstack(df_params)
                if metrics_path.exists() and metrics_path.stat().st_size > 12:
                    try:
                        df_metrics = pl.read_parquet(metrics_path)
                    except Exception as e:
                        print(f"Failed to read metrics parquet for run_id {run_id}")
                        continue
                    df_metrics = df_metrics.select(pl.all().implode())
                    df_run = df_run.hstack(df_metrics)
                stats_l.append(df_run)
                
            df = pl.concat(stats_l, how="diagonal_relaxed").sort(pl.col("run_id"))
            if refresh:
                try:
                    df.write_parquet(self.exp_path / Path(fname))
                    df_nonest = df.pipe(self._resolve_nested)
                    df_nonest.write_csv(self.exp_path / Path(self.results_csv))
                    return df
                except Exception as e:
                    print(f"Failed to write results parquet at {self.exp_path / Path(fname)}")
            else:
                if (self.exp_path / Path(fname)).stat().st_size > 12:
                    try:
                        pl.read_parquet(self.exp_path / Path(fname))
                    except Exception as e:
                        print(f"Failed to read results parquet at {self.exp_path / Path(fname)}")
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

class RunViewer(RunManager):
    def __init__(self, exp_path):
        """
        Args:
             exp_path: 実験ディレクトリのパス (str or Path)
             ex) "./exp_tpls/exp_default"
        """
        self.exp_path = Path(exp_path).resolve()
        self.runs_path = self.exp_path / "runs"
        
        self.run_id = None
        self.run_path = None
        self.df_params = None
        self.df_metrics = None
        
        if not self.exp_path.exists():
            raise FileNotFoundError(f"Experiment path not found: {self.exp_path}")

    def fetch_results(self, fname=None, refresh=False):
        return self.ref_results(fname=fname, refresh=refresh)

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
            df = RunViewer(exp_path=exp_path).fetch_results(refresh=refresh)
            df = df.with_columns(pl.lit(exp_name).alias("exp_name"))
            df = df.select(["exp_name", *df.columns[:-1]])
            dfs.append(df)
            
    dfs_cat = pl.concat(dfs, how="diagonal_relaxed")
    
    return dfs_cat

