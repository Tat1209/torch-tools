import os
import time
import fcntl
import itertools
import multiprocessing
import traceback
import pynvml
import logging
from typing import Any, IO, Callable, Iterator
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
from queue import Empty

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Data Structures ---
@dataclass
class TaskResult:
    """
    タスクの実行結果を保持するデータクラス
    """
    success: bool
    gpu_id: int
    config: Any
    error_msg: str | None = None
    elapsed_time: float = 0.0

def generate_tasks_grid(task_func: Callable, config: dict[str | tuple[str, ...], Any]) -> list[tuple[Callable, dict[str, Any]]]:
    """
    設定辞書から直積（Grid Search）または同期（Zip）させたタスクリストを生成します。

    Args:
        task_func (Callable): 実行する関数。
        config (dict): パラメータ辞書。キーをタプルにすると値のリストがZip同期されます。

    Returns:
        list[tuple[Callable, dict]]: (関数, 設定辞書) のタプルリスト。

    Example:
        >>> def train(args): print(args)
        >>> conf = {
        ...     "bs": [16, 32],                       # Grid: 2通り
        ...     ("model", "lr"): [("A", 0.1), ("B", 0.01)] # Zip: 2通り
        ... }
        >>> tasks = generate_tasks_grid(train, conf)
        >>> # Result: 4 tasks
    """
    sweep_items = []
    fixed = {}

    for k, v in config.items():
        if isinstance(v, list):
            sweep_items.append((k, v))
        else:
            fixed[k] = v

    if not sweep_items:
        return [(task_func, deepcopy(fixed))]

    keys, values_list = zip(*sweep_items)
    tasks = []

    for combo in itertools.product(*values_list):
        new_cfg = deepcopy(fixed)
        for key_def, val in zip(keys, combo):
            if isinstance(key_def, tuple):
                for sub_k, sub_v in zip(key_def, val):
                    new_cfg[sub_k] = sub_v
            else:
                new_cfg[key_def] = val
        tasks.append((task_func, new_cfg))

    return tasks

def get_device(
    avoid_used: bool,
    util_th: int,
    free_mem_th: float,
    avoid_locked: bool = True,
    lock_path: Path = Path("~/.gpu_locks"),
    lock: bool = True,
    gpu_ids: list[int] | None = None
) -> tuple[Any, IO] | tuple[None, None]:
    """
    空いているGPUを探し、ファイルロックを取得して返します。

    Args:
        avoid_used (bool): TrueならGPU使用率・メモリも監視する。
        util_th (int): 使用率閾値(%)。これ以下なら選択。
        free_mem_th (float): 空きメモリ閾値(MiB)。これ以上なら選択。
        avoid_locked (bool): TrueならロックファイルがあるGPUを避ける。
        lock_path (Path): ロックファイルの保存先。
        lock (bool): Trueならロックを取得して返す。
        gpu_ids (list[int]): 使用許可するGPU IDリスト。

    Returns:
        tuple: (Deviceオブジェクト, ロックファイルオブジェクト) または (None, None)
    """
    lock_path = lock_path.expanduser()
    lock_path.mkdir(parents=True, exist_ok=True)

    try:
        pynvml.nvmlInit()
        
        if gpu_ids is not None:
            candidates = gpu_ids
        else:
            candidates = range(pynvml.nvmlDeviceGetCount())
        
        for i in candidates:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            except pynvml.NVMLError:
                continue
            
            if avoid_used:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                free_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).free / (1024 ** 2)

                if util > util_th or free_mem < free_mem_th:
                    continue

            lock_file = lock_path / f"gpu_{i}.lock"
            
            if avoid_locked and lock_file.exists():
                try:
                    with open(lock_file, "w") as f:
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(f, fcntl.LOCK_UN)
                except OSError:
                    continue

            if lock:
                try:
                    f = open(lock_file, "w")
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    Device = type("Device", (object,), {"index": i})
                    return Device(), f
                except OSError:
                    f.close()
                    continue
            else:
                Device = type("Device", (object,), {"index": i})
                return Device(), None

    except pynvml.NVMLError:
        return None, None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return None, None

def _worker_wrapper(
    task_func: Callable, 
    args: Any, 
    gpu_id: int, 
    result_queue: multiprocessing.Queue
):
    """
    [内部関数] ワーカープロセス内でタスクを実行し、結果を送信する。
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    p_name = multiprocessing.current_process().name
    start_time = time.time()
    success = False
    error_msg = None

    try:
        logger.info(f"[{p_name}] Start Task on GPU {gpu_id}")
        task_func(args)
        success = True
        logger.info(f"[{p_name}] Done Task on GPU {gpu_id}")
    except Exception:
        error_msg = traceback.format_exc()
        logger.error(f"[{p_name}] Task failed on GPU {gpu_id}\n{error_msg}")
    finally:
        elapsed = time.time() - start_time
        result = TaskResult(
            success=success,
            gpu_id=gpu_id,
            config=args,
            error_msg=error_msg,
            elapsed_time=elapsed
        )
        result_queue.put(result)

def _spawn_worker(
    ctx: multiprocessing.context.BaseContext,
    task_func: Callable, 
    args: Any, 
    gpu_id: int, 
    result_queue: multiprocessing.Queue
) -> multiprocessing.Process:
    """
    [内部関数] 指定されたコンテキスト(spawn)でワーカープロセスを起動する。
    """
    p = ctx.Process(
        target=_worker_wrapper, 
        args=(task_func, args, gpu_id, result_queue),
        name=f"Worker-GPU{gpu_id}"
    )
    p.start()
    return p

def _cleanup_finished_processes(running_procs: dict[multiprocessing.Process, IO]):
    """
    [内部関数] 終了したプロセスを整理し、ロックを解放する。
    """
    for p in list(running_procs.keys()):
        if not p.is_alive():
            p.join()
            lock_handle = running_procs[p]
            if lock_handle:
                try:
                    fcntl.flock(lock_handle, fcntl.LOCK_UN)
                    lock_handle.close()
                except IOError as e:
                    logger.warning(f"Failed to close lock: {e}")
            del running_procs[p]

def parallel_run(
    tasks: list[tuple[Callable, Any]] | Iterator[tuple[Callable, Any]],
    max_tasks: int | None = None,
    check_interval: float = 10.0,
    avoid_used: bool = False,
    util_th: int = 10,
    free_mem_th: float = 2000.0,
    lock_path: Path = Path("~/.gpu_locks"),
    gpu_ids: list[int] | None = None
) -> list[TaskResult]:
    """
    GPUリソースを管理しながらタスクを並列実行するスケジューラ。

    Args:
        tasks (list | Iterator): (関数, 引数) のタプルリスト。
        max_tasks (int | None): 最大同時実行プロセス数。
        check_interval (float): 空きGPU探索の間隔(秒)。
        avoid_used (bool): GPU負荷(util/mem)監視を行うか。
        util_th (int): 使用率上限(%)。
        free_mem_th (float): 必要空きメモリ(MiB)。
        lock_path (Path): ロックファイル保存パス。
        gpu_ids (list[int]): 使用するGPU ID。

    Returns:
        list[TaskResult]: 全タスクの実行結果。

    Example:
        >>> tasks = generate_tasks_grid(my_func, config)
        >>> results = parallel_run(tasks, max_tasks=4, gpu_ids=[0, 1, 2, 3])
    """
    task_list = list(tasks)
    total_tasks = len(task_list)
    logger.info(f"[Scheduler] Total tasks scheduled: {total_tasks}")
    
    running_procs: dict[multiprocessing.Process, IO] = {}
    
    # 【重要】spawnコンテキストでQueueとProcessを生成する（SemLockエラー対策）
    ctx = multiprocessing.get_context('spawn')
    result_queue = ctx.Queue()
    
    results: list[TaskResult] = []
    task_idx = 0

    try:
        while task_idx < total_tasks or running_procs:
            _cleanup_finished_processes(running_procs)

            while not result_queue.empty():
                try:
                    results.append(result_queue.get_nowait())
                except Empty:
                    break

            if task_idx < total_tasks and (max_tasks is None or len(running_procs) < max_tasks):
                res = get_device(
                    avoid_used=avoid_used, util_th=util_th, free_mem_th=free_mem_th,
                    avoid_locked=True, lock_path=lock_path, lock=True, gpu_ids=gpu_ids
                )
                
                if res and res[0] is not None:
                    device, lock_handle = res
                    gpu_id = device.index
                    
                    func, args = task_list[task_idx]
                    p = _spawn_worker(ctx, func, args, gpu_id, result_queue)
                    running_procs[p] = lock_handle
                    
                    task_idx += 1
                    logger.info(f"[Scheduler] Assigned GPU {gpu_id}. Progress: {task_idx}/{total_tasks} (Running: {len(running_procs)})")
                else:
                    time.sleep(check_interval)
            else:
                if running_procs:
                    time.sleep(check_interval)

        logger.info("[Scheduler] All tasks completed.")

    except KeyboardInterrupt:
        logger.warning("\n[Scheduler] Terminating...")
        for p in list(running_procs.keys()):
            if p.is_alive():
                p.terminate()
                p.join()
                lh = running_procs[p]
                if lh:
                    try:
                        fcntl.flock(lh, fcntl.LOCK_UN)
                        lh.close()
                    except Exception:
                        pass
        logger.info("[Scheduler] Shutdown complete.")
        return results

    failed_tasks = [r for r in results if not r.success]
    if failed_tasks:
        logger.error(f"--- {len(failed_tasks)} Tasks Failed ---")
        for f in failed_tasks:
            logger.error(f"Config: {f.config}\nError: {f.error_msg}\n")
    else:
        logger.info("--- All Tasks Succeeded ---")

    return results