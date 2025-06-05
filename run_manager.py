from pathlib import Path

import polars as pl
from polars.exceptions import PolarsError

import utils


class PathManager:
    def __init__(self, exc_path=None, exp_name="exp_default", exp_path=None):
        # exp_pathを指定 -> 格納ディレクトリとexp_nameを同時に指定
        if exc_path is None:
            exp_path = Path(exp_path).resolve()
            pa_path = exp_path.parent

        # exc_path(__file__)とexp_nameを指定 -> コード実行ディレクトリと同じディレクトリに結果を格納
        else:
            pa_path = Path(exc_path).resolve().parent  # 常に__file__が送られてくるならresolveは不要
            exp_path = pa_path / Path(exp_name)

        self.pa_path = pa_path
        self.exp_path = exp_path
        self.runs_path = exp_path / Path("runs")
        self.run_path = None
        self.run_id = None

    def fpath(self, fname):
        return self.run_path / Path(fname)
    

class RunManager(PathManager):
    def __init__(self, exc_path=None, exp_name="exp_default", exp_path=None, run_id=None):
        """
        ex.)
            run = RunManager(exc_path=__file__, exp_name="exp_nyancat")
        """
        super().__init__(exc_path, exp_name, exp_path)

        # 複数同時に作ったときに残らないから却下
        # if auto_delete  and  exp_name == "exp_tmp":
        #     if Path(self.exp_path).exists():
        #         shutil.rmtree(self.exp_path)
        
        self.exp_path.mkdir(parents=True, exist_ok=True)
        self.runs_path.mkdir(parents=True, exist_ok=True)

        if run_id is None:
            run_id = self._get_run_id(self.runs_path)
        else:
            self._inherit_stats()

        run_path = self.runs_path / str(run_id)
        run_path.mkdir(parents=True, exist_ok=True)

        self.run_path = run_path
        self.run_id = run_id
        
        self.df_params = None
        self.df_metrics = None
        
        self.params_name = "_params.parquet"
        self.metrics_name = "_metrics.parquet"
        self.stats_name = "_stats.parquet"
        
    def exe_path(self, func, fname):
        path = self.fpath(fname)
        func(path)

    def _get_run_id(self, runs_path):
        dir_names = list(runs_path.iterdir())
        dir_nums = [int(dir_name.name) for dir_name in dir_names]
        if len(dir_nums) == 0:
            run_id = 0
        else:
            run_id = max(dir_nums) + 1
        return run_id

    def log_text(self, fname, text):
        with open(self.fpath(fname), "w") as fh:
            fh.write(text)

    def log_torch_save(self, object, fname):
        import torch
        torch.save(object, self.fpath(fname))
        
    def log_param(self, name, value):
        self.log_params({name: value})

    def log_params(self, stored_dict):
        stored_dict = {k: [v] for k, v in stored_dict.items()}
        if self.df_params is None:
            self.df_params = pl.DataFrame(stored_dict)
        else:
            self.df_params = self.df_params.hstack(pl.DataFrame(stored_dict))

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
        if self.fpath(self.params_name).exists():
            self.df_params = pl.read_parquet(self.fpath(self.params_name))
        if self.fpath(self.metrics_name).exists():
            self.df_metrics = pl.read_parquet(self.fpath(self.metrics_name))

    def _fetch_stats(self):
            if self.df_params is not None and self.df_metrics is not None:
                return self.df_params.hstack(self.df_metrics[-1].select(pl.all().exclude("step")))

            elif self.df_params is not None:
                return self.df_params

            elif self.df_metrics is not None:
                return self.df_metrics[-1].select(pl.all().exclude("step"))

            else:
                return pl.DataFrame()

    def ref_stats(self, step=None, itv=None, last_step=None):
        if utils.interval(step=step, itv=itv, last_step=last_step):
            if self.df_params is not None:
                self.df_params.write_parquet(self.fpath(self.params_name))
                df_nonest = self._resolve_nested(self.df_params)
                df_nonest.write_csv(self.fpath("_params.csv"))
            if self.df_metrics is not None:
                self.df_metrics.write_parquet(self.fpath(self.metrics_name))
                df_nonest = self._resolve_nested(self.df_metrics)
                df_nonest.write_csv(self.fpath("_metrics.csv"))
            df_stats = self._fetch_stats()
            df_stats.write_parquet(self.fpath(self.stats_name))
            df_nonest = self._resolve_nested(self.df_metrics)
            df_nonest.write_csv(self.fpath("_stats.csv"))
            
    def _resolve_nested(self, df):
        nested_columns = [name for name, dtype in zip(df.columns, df.dtypes) if dtype.is_nested()]

        # nested_types = (pl.List, pl.Struct)
        # nested_columns = [name for name, dtype in df.schema.items() if isinstance(dtype, nested_types)]
        for name in nested_columns:
            try:
                # df = df.with_columns([pl.col(name).list.mean().alias(name) for name in nested_columns])
                df = df.with_columns([(pl.lit("mean: ") + pl.col(name).list.mean().cast(pl.Utf8)).alias(name)])
            except PolarsError:
                # df = df.with_columns([pl.col(name).list.last().alias(name) for name in nested_columns])
                df = df.with_columns([(pl.lit("last: ") + pl.col(name).list.last().cast(pl.Utf8)).alias(name)])
                
        return df

    def fetch_files(self, fname):
        dir_names = list(self.runs_path.iterdir())
        run_ids = [int(dir_name.name) for dir_name in dir_names]
        stats_paths = [dir_name / Path(fname) for dir_name in dir_names]
        
        return run_ids, stats_paths
            
    def ref_results(self, step=None, itv=None, last_step=None, fname="results.parquet", refresh=True, met_listed=False):
        if utils.interval(step=step, itv=itv, last_step=last_step):
            run_ids, params_paths = self.fetch_files(self.params_name)
            run_ids, metrics_paths = self.fetch_files(self.metrics_name)

            stats_l = []
            for run_id, params_path, metrics_path in zip(run_ids, params_paths, metrics_paths):
                df_run = pl.DataFrame({"run_id": run_id})
                if params_path.exists():
                    df_params = pl.read_parquet(params_path)
                    df_run = df_run.hstack(df_params)
                if metrics_path.exists():
                    df_metrics = pl.read_parquet(metrics_path)
                    if met_listed:
                        df_metrics = df_metrics.select(pl.all().implode())
                    else:
                        df_metrics = df_metrics[-1].select(pl.all().exclude("step"))
                    df_run = df_run.hstack(df_metrics)
                stats_l.append(df_run)
                
            df = pl.concat(stats_l, how="diagonal_relaxed").sort(pl.col("run_id"))
            # if refresh  or  not met_listed:
                # df = df.with_columns(pl.col([col for col, dtype in df.schema.items() if dtype.is_nested()]).list.last().name.keep()) 
            if refresh:
                df.write_parquet(self.exp_path / Path(fname))
                df_nonest = self._resolve_nested(df)
                df_nonest.write_csv(self.exp_path / Path("results.csv"))
            return df

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


class RunViewer(RunManager, PathManager):
    def __init__(self, exc_path=None, exp_name="exp_default", exp_path=None, run_id=None):
        PathManager.__init__(self, exc_path, exp_name, exp_path)

        # 応急処置
        # RunManagerのmkdir周りを最初の格納時に行い、init周りを整理
        self.params_name = "_params.parquet"
        self.metrics_name = "_metrics.parquet"
        self.stats_name = "_stats.parquet"
        
    def fetch_results(self, fname="results.parquet", refresh=True, met_listed=False):
        return self.ref_results(fname=fname, refresh=refresh, met_listed=met_listed)

    # def _fetch_metrics(self, listed=False):
    #     run_ids, stats_paths = self.fetch_files("metrics.csv")

    #     stats_l = []
    #     for run_id, stats_path in zip(run_ids, stats_paths):
    #         try:
    #             df_stats = pl.read_csv(stats_path)
    #             df_stats_wid = df_stats.with_columns(pl.lit(run_id).alias("run_id"))
    #             df_stats_wid = df_stats_wid.select(["run_id"] + df_stats.columns)
    #             stats_l.append(df_stats_wid)
    #         except FileNotFoundError:
    #             # metrics.csvがない場合Errorが発生する。
    #             pass
    #     if listed:
    #         return stats_l
    #     else:
    #         df = pl.concat(stats_l, how="diagonal_relaxed").sort(pl.col("step")).sort(pl.col("run_id"))
    #         return df
        
        
