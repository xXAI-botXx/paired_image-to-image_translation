"""
# Runtime Guard

This module provides utilities to **monitor hardware usage** and **prevent server stalls**
during long-running loops, training jobs, or other iterative processes.

## Features

- RAM usage monitoring
- CPU usage monitoring
- GPU memory monitoring (PyTorch / TensorFlow)
- Mean loop time tracking
- Watchdog timer
- Python memory leak detection

## Usage Patterns

### Standard:
```python
import runtime_guard as run

guard = run.RuntimeGuard()

for epoch in epochs:
    for idx, x, y in data:
        guard.start_loop()

        # do work

        guard.update()

    if validate:
        guard.pause()
        # validation
```

### Iterator-Method:
```python
import runtime_guard as run

guard = run.RuntimeGuard()

for epoch in epochs:
    for idx, x, y in guard(data):
        # do work

    with run.Pause(guard):
        if validate:
            # or just: guard.pause()
            # validation
```

### Context-Method:
```python
import runtime_guard as run

guard = run.RuntimeGuard()

for epoch in epochs:
    for idx, x, y in data:
        with run.Run(guard):
            # do work

    with run.Pause(guard):
        if validate:
            # or just: guard.pause()
            # validation
```
"""
# =============
# || Imports ||
# =============

# RAM Check
import psutil
import os
import time
import datetime

# GPU Check
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import tensorflow as tf
    _HAS_TF = True
except ImportError:
    _HAS_TF = False

# Mean Loop Time Check
from collections import deque

# Watchdog
import threading

# Leak Check
# import tracemalloc
# import gc
# import sys
# import types
# import heapq  # efficient top-k items
import tracemalloc


# =============
# || Helpers ||
# =============
def add_custom_args(parser):
    """
    ## Add RuntimeGuard CLI Arguments

    Extends an `argparse.ArgumentParser` with RuntimeGuard-related options.

    ### Parameters
    - **parser** (`argparse.ArgumentParser`): Parser instance to extend

    ### Returns
    - `argparse.ArgumentParser`: Updated parser
    """
    # parser = argparse.ArgumentParser()
    parser.add_argument('--use_runtime_guard', action='store_true', help='Whether to use a runtime guardian')
    parser.add_argument('--runtime_guard_make_ram_check', action='store_true')
    parser.add_argument('--runtime_guard_max_ram_usage_percentage', type=float, default=0.9)
    parser.add_argument('--runtime_guard_make_cpu_check', action='store_true')
    parser.add_argument('--runtime_guard_max_cpu_usage', type=float, default=0.9)
    parser.add_argument('--runtime_guard_make_gpu_check', action='store_true')
    parser.add_argument('--runtime_guard_max_gpu_usage', type=float, default=0.9)
    parser.add_argument('--runtime_guard_make_mean_loop_time_check', action='store_true')
    parser.add_argument('--runtime_guard_max_duration_factor_percentage', type=float, default=3.0)
    parser.add_argument('--runtime_guard_goal_mean_loop_time', type=float, default=None)
    parser.add_argument('--runtime_guard_mean_loop_time_window_size', type=int, default=50)
    parser.add_argument('--runtime_guard_make_watchdog_timer_check', action='store_true')
    parser.add_argument('--runtime_guard_max_watchdog_seconds_timeout', type=int, default=300)
    parser.add_argument('--runtime_guard_watchdog_seconds_waittime', type=int, default=30)
    parser.add_argument('--runtime_guard_make_leak_check', action='store_true')
    parser.add_argument('--runtime_guard_max_leak_mb', type=float, default=200.0)
    parser.add_argument('--runtime_guard_max_leak_ratio', type=float, default=0.2)
    parser.add_argument('--runtime_guard_should_print', action='store_true')
    parser.add_argument('--runtime_guard_print_every_x_calls', type=int, default=5000)
    parser.add_argument('--runtime_guard_should_log', action='store_true')
    parser.add_argument('--runtime_guard_log_every_x_calls', type=int, default=5000)
    parser.add_argument('--runtime_guard_log_path', type=str, default="./")
    parser.add_argument('--runtime_guard_warm_up_iter', type=int, default=50)
    parser.add_argument('--runtime_guard_update_every_x_calls', type=int, default=50)
    parser.add_argument('--runtime_guard_hard_exit', action='store_true')

    # new_parse_env = parser.parse_args(namespace=parse_env)

    return parser

def getDummyGuard():
    """
    ## Dummy RuntimeGuard

    Creates a `RuntimeGuard` instance with **all checks disabled**.

    Useful for:
    - Testing
    - Debug runs
    - Disabling monitoring without code changes

    ### Returns
    - `RuntimeGuard`: Guard instance with disabled checks
    """
    return RuntimeGuard(make_ram_check=False, 
                        max_ram_usage_percentage=2.0,
                        make_cpu_check=False, max_cpu_usage=2.0, 
                        make_gpu_check=False, max_gpu_usage=2.0,
                        make_mean_loop_time_check=False, max_duration_factor_percentage=3.0, goal_mean_loop_time=None, mean_loop_time_window_size=50,
                        make_watchdog_timer_check=False, max_watchdog_seconds_timeout=60*5, watchdog_seconds_waittime=30, # 5 minutes
                        make_leak_check=False, max_leak_mb=100, max_leak_ratio=0.2,
                        should_print=False, print_every_x_calls=9999999,
                        should_log=False, log_every_x_calls=9999999,
                        warm_up_iter=0, update_every_x_calls=50,
                        hard_exit=False)

# ================
# || Main Class ||
# ================
# TrainGuard
class RuntimeGuard(object):
    """
    ## RuntimeGuard

    Monitors system resources and loop performance to prevent:
    - Server freezes
    - Memory exhaustion
    - GPU overcommit
    - Silent training stalls

    ### Supported Checks
    - RAM usage
    - CPU load
    - GPU memory
    - Mean loop duration
    - Watchdog heartbeat
    - Python memory leaks

    ### Capabilities
    - Pause / resume monitoring
    - Console printing
    - File logging
    - Soft or hard exit on failure
    """
    def __init__(self, make_ram_check=True, max_ram_usage_percentage=0.9,
                       make_cpu_check=True, max_cpu_usage=0.9, 
                       make_gpu_check=True, max_gpu_usage=0.9, 
                       make_mean_loop_time_check=True, max_duration_factor_percentage=3.0, goal_mean_loop_time=None, mean_loop_time_window_size=50,
                       make_watchdog_timer_check=True, max_watchdog_seconds_timeout=60*5, watchdog_seconds_waittime=30, # 5 minutes
                       make_leak_check=True, max_leak_mb=100, max_leak_ratio=0.2,
                       should_print=False, print_every_x_calls=5000,
                       should_log=True, log_every_x_calls=5000, log_path="./runtime_guard.log",
                       warm_up_iter=50, update_every_x_calls=50,
                       hard_exit=False):
        """
        ## Initialize RuntimeGuard

        Configures which runtime checks are active and their thresholds.

        ### Parameters
        - **make_ram_check** (`bool`): Enable RAM monitoring
        - **max_ram_usage_percentage** (`float`): Max RAM usage (fraction of total)
        - **make_cpu_check** (`bool`): Enable CPU monitoring
        - **max_cpu_usage** (`float`): Max CPU usage (fraction)
        - **make_gpu_check** (`bool`): Enable GPU memory monitoring
        - **max_gpu_usage** (`float`): Max GPU memory usage (fraction)
        - **make_mean_loop_time_check** (`bool`): Enable loop-time monitoring
        - **max_duration_factor_percentage** (`float`): Allowed slowdown factor
        - **goal_mean_loop_time** (`float | None`): Reference loop duration
        - **mean_loop_time_window_size** (`int`): Moving average window
        - **make_watchdog_timer_check** (`bool`): Enable watchdog
        - **max_watchdog_seconds_timeout** (`int`): Timeout before abort
        - **watchdog_seconds_waittime** (`int`): Watchdog polling interval
        - **make_leak_check** (`bool`): Enable memory leak detection
        - **max_leak_mb** (`float`): Max allowed memory growth
        - **max_leak_ratio** (`float`): Max relative growth
        - **should_print** (`bool`): Enable console output
        - **print_every_x_calls** (`int`): Print interval
        - **should_log** (`bool`): Enable file logging
        - **log_every_x_calls** (`int`): Log interval
        - **log_path** (`str`): Log file path
        - **warm_up_iter** (`int`): Warm-up iterations
        - **update_every_x_calls** (`int`): Check frequency
        - **hard_exit** (`bool`): Force `os._exit()` on failure
        """
        self.make_ram_check = make_ram_check
        self.max_ram_gb = max_ram_usage_percentage * psutil.virtual_memory().total / (1024**3)
        
        self.make_cpu_check = make_cpu_check
        self.max_cpu_usage = max_cpu_usage

        self.make_gpu_check = make_gpu_check
        self.max_gpu_usage = max_gpu_usage

        self.make_mean_loop_time_check = make_mean_loop_time_check
        self.max_duration_factor_percentage = max_duration_factor_percentage
        self.goal_mean_loop_time_collected = True if goal_mean_loop_time else False
        self.mean_loop_time_window_size = mean_loop_time_window_size
        self.goal_mean_loop_time_counter = 0
        self.goal_mean_loop_time = goal_mean_loop_time if goal_mean_loop_time else 0.0
        self.mean_loop_time_window = deque(maxlen=self.mean_loop_time_window_size)
        self.cur_loop_start_time = None

        self.make_watchdog_timer_check = make_watchdog_timer_check
        self.max_watchdog_seconds_timeout = max_watchdog_seconds_timeout
        self.watchdog_seconds_waittime = watchdog_seconds_waittime
        self.last_heartbeat = time.time()
        self.watchdog_should_wait = False
        if make_watchdog_timer_check:
            self.check_watchdog_timer()
        self.abort_event = threading.Event()

        self.make_leak_check = make_leak_check
        self.max_leak_mb = max_leak_mb
        self.max_leak_ratio = max_leak_ratio
        self.last_snapshot = None
        if make_leak_check:
            tracemalloc.start()

        self.should_print = should_print
        self.print_every_x_calls = print_every_x_calls
        self.should_log = should_log
        self.log_every_x_calls = log_every_x_calls
        self.log_path = log_path
        self.warm_up_iter = warm_up_iter
        self.update_every_x_calls = update_every_x_calls
        self.cur_call_n = 0
        self.hard_exit = hard_exit

        if should_log:
            with open(log_path, "w") as file_:
                file_.write(str(self)+"\n")

        self.iteration_setting_check()

    def __str__(self):
        str_repr = "Runtime Guard Configuration\n"
        str_repr += f"- RAM Check: {self.make_ram_check}"
        if self.make_ram_check:
            str_repr += f" (max {self.max_ram_gb:.2f} GB)"

        str_repr += f"\n- CPU Check: {self.make_cpu_check}"
        if self.make_cpu_check:
            str_repr += f" (max usage {self.max_cpu_usage*100:.1f}%)"

        str_repr += f"\n- GPU Check: {self.make_gpu_check}"
        if self.make_gpu_check:
            str_repr += f" (max usage {self.max_gpu_usage*100:.1f}%)"

        str_repr += f"\n- Mean Loop Time Check: {self.make_mean_loop_time_check}"
        if self.make_mean_loop_time_check:
            str_repr += f"\n    -> max_duration_factor_percentage = {self.max_duration_factor_percentage}"
            str_repr += f"\n    -> goal_mean_loop_time = {self.goal_mean_loop_time}"
            str_repr += f"\n    -> mean_loop_time_window_size = {self.mean_loop_time_window_size}"

        str_repr += f"\n- Watchdog Timer Check: {self.make_watchdog_timer_check}"
        if self.make_watchdog_timer_check:
            str_repr += f"\n    -> max_watchdog_seconds_timeout = {self.max_watchdog_seconds_timeout}"
            str_repr += f"\n    -> watchdog_seconds_waittime = {self.watchdog_seconds_waittime}"

        str_repr += f"\n- Memory Leak Check: {self.make_leak_check}"
        if self.make_leak_check:
            str_repr += f"\n    -> max_leak_mb = {self.max_leak_mb}"
            str_repr += f"\n    -> max_leak_ratio = {self.max_leak_ratio}"

        str_repr += f"\n- Print enabled: {self.should_print}"
        if self.should_print:
            str_repr += f" (every {self.print_every_x_calls} calls)"

        str_repr += f"\n- Logging enabled: {self.should_log}"
        if self.should_log:
            str_repr += f" (every {self.log_every_x_calls} calls, path='{self.log_path}')"

        str_repr += f"\n- Warm-up iterations: {self.warm_up_iter}"
        str_repr += f"\n- Update every {self.update_every_x_calls} calls"
        str_repr += f"\n- Hard exit on failure: {self.hard_exit}"

        return str_repr
    
    def __iter__(self):
        if hasattr(self, "iterable"):
            for item in self.iterable:
                self.start_loop()
                try:
                    yield item
                finally:
                    self.update()
    
    def __call__(self, iterable):
        self.iterable = iterable
        return iter(self)

    def update(self):
        """
        ## Update RuntimeGuard

        Executes all enabled checks:
        - RAM
        - CPU
        - GPU
        - Mean loop time
        - Watchdog
        - Memory leak detection

        Handles:
        - Printing
        - Logging
        - Abort conditions
        """
        if self.cur_call_n < self.warm_up_iter or \
            self.cur_call_n % self.update_every_x_calls != 0:
            self.last_heartbeat = time.time()
            self.cur_call_n += 1
            return

        # make checks
        if self.make_ram_check:
            ram_result = self.check_ram()
        else:
            ram_result = "    - No RAM Checking"

        if self.make_cpu_check:
            cpu_result = self.check_cpu()
        else:
            cpu_result = "    - No CPU Checking"

        if self.make_gpu_check:
            gpu_result = self.check_gpu()
        else:
            gpu_result = "    - No GPU Checking"

        if self.make_mean_loop_time_check:
            self.next_loop()
            mean_loop_time_result = self.check_mean_loop_time()
        else:
            mean_loop_time_result = "    - No Mean Loop Time Checking"

        if self.abort_event.is_set():
            self.exit(msg="[ABORT] Watchdog timeout")
        else:
            watchdog_result = "    - Watchdog is still fine!"
        
        if self.make_leak_check:
            leak_result = self.check_leak()
        else:
            leak_result = "    - No Leak Checking"

        # printing and logging
        self.make_print_entry(ram_result=ram_result, cpu_result=cpu_result, 
                              gpu_result=gpu_result, mean_loop_time_result=mean_loop_time_result, 
                              watchdog_result=watchdog_result, leak_result=leak_result)
        self.make_log_entry(ram_result=ram_result, cpu_result=cpu_result, 
                            gpu_result=gpu_result, mean_loop_time_result=mean_loop_time_result, 
                            watchdog_result=watchdog_result, leak_result=leak_result)
        
        # update vars
        self.cur_call_n += 1
        self.last_heartbeat = time.time()

    def start_loop(self):
        """
        ## Start Loop Iteration

        - Resets loop timer
        - Sends watchdog heartbeat
        """
        self.reset_timer()
        self.continue_watchdog()

    def pause(self):
        """
        ## Pause Watchdog

        Temporarily disables watchdog checks.

        Useful during:
        - Validation
        - Blocking I/O
        - Long evaluations
        """
        self.pause_watchdog()

    def check_ram(self):
        """
        ## Resource Check

        Performs the respective resource check and aborts if thresholds are exceeded.

        ### Returns
        - `str`: Human-readable status message
        """
        used_gb = psutil.virtual_memory().used / (1024**3)
        if used_gb > self.max_ram_gb:
            abort_txt = f"[ABORT] RAM usage too high: {used_gb:.2f} GB"
            self.make_abort_log(abort_message=abort_txt)
            print(abort_txt)
            self.exit(msg=abort_txt)

        return f"    - RAM checked\n        -> usage = {used_gb:.2f} GB / {self.max_ram_gb:.2f} GB ({(used_gb/self.max_ram_gb)*100:.2f}%)"

    def check_cpu(self):
        """
        ## Resource Check

        Performs the respective resource check and aborts if thresholds are exceeded.

        ### Returns
        - `str`: Human-readable status message
        """
        # cpu_usage = psutil.cpu_percent(interval=None) / 100.0
        cpu_usage = psutil.getloadavg()[0] / os.cpu_count()

        if cpu_usage > self.max_cpu_usage:
            abort_txt = f"[ABORT] CPU usage too high: {cpu_usage*100:.2f}%"
            self.make_abort_log(abort_message=abort_txt)
            print(abort_txt)
            self.exit(msg=abort_txt)

        return f"    - CPU checked\n        -> usage = {cpu_usage*100:.2f}%"

    def check_gpu(self):
        """
        ## Resource Check

        Performs the respective resource check and aborts if thresholds are exceeded.

        ### Returns
        - `str`: Human-readable status message
        """
        if _HAS_TORCH and torch.cuda.is_available():
            used = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            usage_ratio = used / total
            if usage_ratio > self.max_gpu_usage:
                abort_txt = f"[ABORT] GPU memory full (PyTorch)"
                self.make_abort_log(abort_message=abort_txt)
                print(abort_txt)
                self.exit(msg=abort_txt)
            return f"    - GPU checked (PyTorch)\n        -> usage = {used/(1024**3):.2f} GB / {total/(1024**3):.2f} GB ({usage_ratio*100:.2f}%)"

        elif _HAS_TF:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    details = tf.config.experimental.get_memory_info('GPU:0')
                    used = details['current']
                    total = details['peak']
                    usage_ratio = used / total
                    if usage_ratio > self.max_gpu_usage:
                        abort_txt = f"[ABORT] GPU memory full (TensorFlow)"
                        self.make_abort_log(abort_message=abort_txt)
                        print(abort_txt)
                        self.exit(msg=abort_txt)
                    return f"    - GPU checked (TensorFlow)\n        -> usage = {used/(1024**3):.2f} GB / {total/(1024**3):.2f} GB ({usage_ratio*100:.2f}%)"
                except Exception as e:
                    return f"    - GPU check (TensorFlow) failed: {e}"
            else:
                return "    - GPU checked (TensorFlow) -> no GPU found"

        else:
            return "    - GPU check skipped (no supported framework installed)"


    def check_mean_loop_time(self):
        """
        ## Mean Loop Time Check

        Compares current mean loop duration against the baseline.

        Aborts if slowdown exceeds the configured factor.

        ### Returns
        - `str`: Status report
        """
        if not self.goal_mean_loop_time_collected:
            return "    - Mean Loop Time not yet collected"
        
        cur_mean_time = self.get_mean_loop_time()
        if cur_mean_time <= 0:
            return "    - Mean Loop Time not yet collected"
        
        duration_factor_percentage = cur_mean_time / self.goal_mean_loop_time
        if duration_factor_percentage > self.max_duration_factor_percentage:
            abort_txt = f"[ABORT] Process got slowed down by {duration_factor_percentage} compared to the beginning"
            self.make_abort_log(abort_message=abort_txt)
            print(abort_txt)
            self.exit(msg=abort_txt)

        return f"    - Mean Loop Time checked\n        -> mean time trend = {duration_factor_percentage*100:.2f}% compared to the goal loop time\n        -> current mean loop duration {cur_mean_time:.2f} / base mean loop duration {self.goal_mean_loop_time:.2f}"

    def next_loop(self):
        """
        Marks the end of the current loop iteration and updates loop time statistics.

        Tracks duration of the loop, updates the mean loop time window, 
        and sets the goal mean loop time once enough iterations have been collected.
        """
        if self.cur_loop_start_time is None:
            # skip the first loop and start measuring
            self.cur_loop_start_time = time.time()
        else:
            # get time duration
            duration = time.time() - self.cur_loop_start_time

            # add new mean value
            self.mean_loop_time_window.append(duration)

            # update goal mean loop time
            if not self.goal_mean_loop_time_collected and len(self.mean_loop_time_window) >= self.mean_loop_time_window_size:
                self.goal_mean_loop_time = self.get_mean_loop_time()
                self.goal_mean_loop_time_collected = True

            self.cur_loop_start_time = time.time()

    def reset_timer(self):
        """
        Resets the current loop timer and updates the last heartbeat time.

        Useful for marking the start of a new loop iteration.
        """
        self.cur_loop_start_time = time.time()
        self.last_heartbeat = time.time()

    def get_mean_loop_time(self):
        """
        Calculates the current mean loop duration based on the loop time window.

        Returns:
            float: Mean loop duration in seconds. Returns 0.0 if no data collected yet.
        """
        if not self.mean_loop_time_window:
            return 0.0
        return sum(self.mean_loop_time_window) / len(self.mean_loop_time_window)

    def check_watchdog_timer(self):
        """
        ## Watchdog Monitor

        Runs a background daemon thread that:
        - Monitors loop heartbeats
        - Detects stalled execution
        - Triggers abort on timeout
        """
        def watchdog_func():
            start_time = time.time()
            while True:
                time.sleep(self.watchdog_seconds_waittime)
                if not self.watchdog_should_wait:
                    duration = time.time() - self.last_heartbeat
                    if duration > self.max_watchdog_seconds_timeout:
                        abort_txt = f"[ABORT] Training stalled -> Watchdog got delayed by {duration} seconds"
                        self.make_abort_log(abort_message=abort_txt)
                        print(abort_txt)
                        self.abort_event.set()
                        raise RuntimeError(abort_txt)  # will only exit the thread

        # runs alone (no reference to it needed) but get terminated when the pipeline is finish
        self.watchdog = threading.Thread(target=watchdog_func, daemon=True)
        self.watchdog.start()

    def pause_watchdog(self):
        """
        Temporarily pauses the watchdog timer.

        Useful when performing blocking operations like validation,
        where the main loop is intentionally idle.
        """
        self.watchdog_should_wait = True

    def continue_watchdog(self):
        """
        Resumes the watchdog timer and updates the last heartbeat timestamp.

        Should be called at the start of each loop iteration.
        """
        self.watchdog_should_wait = False
        self.last_heartbeat = time.time()

    def check_leak(self):
        """
        ## Memory Leak Detection

        Uses `tracemalloc` snapshots to detect memory growth.

        Aborts if:
        - Absolute growth exceeds limit
        - Relative growth exceeds ratio

        ### Returns
        - `str`: Leak summary
        """
        snapshot = tracemalloc.take_snapshot()

        if self.last_snapshot is None:
            self.last_snapshot = snapshot
            return "    - Leak not checked because there is no last snapshot"

        stats = snapshot.compare_to(self.last_snapshot, 'lineno')

        # current total growth of only increasing memory
        total_growth_bytes = sum(
            stat.size_diff for stat in stats if stat.size_diff > 0
        )

        total_growth_mb = total_growth_bytes / (1024 ** 2)

        # previous total growth of only increasing memory
        prev_total_bytes = sum( stat.size for stat in self.last_snapshot.statistics('lineno') )

        # calc ratio
        if prev_total_bytes > 0:
            growth_ratio = total_growth_bytes / prev_total_bytes
        else:
            growth_ratio = 0.0

        txt = f"\n        -> Growth = {total_growth_mb:.2f} MB"
        txt += f"\n        -> Ratio  = {growth_ratio * 100:.2f}%"
        txt += f"\n        -> Top leak sources:"
        for stat in stats[:10]:
            if stat.size_diff > 0:
                txt += f"\n            {stat}"

        if total_growth_mb > self.max_leak_mb or \
           growth_ratio > self.max_leak_ratio:
            self.last_snapshot = snapshot
            raise RuntimeError(
                f"[ABORT] Python memory leak detected. Details:\n{txt}"
            )

        # No leak -> update snapshot + return txt
        self.last_snapshot = snapshot

        return "    - Leak checked" + txt
    
    # def check_leak(self, top_n=5):
    #     """
    #     Checks for memory growth in Python variables 
    #     compared to the previous snapshot. Raises RuntimeError if growth exceeds thresholds.

    #     Args:
    #         top_n (int): Number of top growing variables to report.

    #     Returns:
    #         str: Status message detailing memory growth.
    #     """
    #     self.make_abort_log("Checkpoint 1: check_leak entered")
    #     # Collect garbage first
    #     if self.cur_call_n % self.gc_every_x_calls == 0:
    #         gc.collect()
        
    #     # Take a snapshot of all current objects (id, repr, type, size)
    #     snapshot = []
    #     for obj in gc.get_objects():
    #         try:
    #             if isinstance(obj, (type, types.ModuleType)):
    #                 continue

    #             obj_repr = "<unrepresentable>"
    #             try:
    #                 obj_repr = repr(obj)
    #             except Exception:
    #                 pass
                
    #             size = 0 
    #             try:
    #                 size = sys.getsizeof(obj)
    #             except Exception:
    #                 pass
                

    #             snapshot += [(id(obj), obj_repr, type(obj), size)]
    #         except Exception:
    #             self.make_abort_log("Got Exception")
    #             continue

    #     self.make_abort_log("Checkpoint 2: objects finish")
        
    #     if not hasattr(self, 'last_var_snapshot') or self.last_var_snapshot is None:
    #         self.last_var_snapshot = snapshot
    #         return "    - Leak not checked because there is no last snapshot"
        
    #     # Create dict mapping object id to previous size
    #     last_dict = {oid: size for oid, _, _, size in self.last_var_snapshot}
        
    #     growth_list = []
    #     for oid, obj_repr, typ, size in snapshot:
    #         prev_size = last_dict.get(oid, 0)
    #         diff = size - prev_size
    #         if diff > 0:
    #             growth_list.append((oid, obj_repr, typ, diff))
        
    #     # Sort by descending growth
    #     growth_list.sort(key=lambda x: x[3], reverse=True)
        
    #     # Total growth
    #     total_growth_bytes = sum(diff for _, _, _, diff in growth_list)
    #     total_growth_mb = total_growth_bytes / (1024 ** 2)
        
    #     # Previous total size
    #     prev_total_bytes = sum(size for _, _, _, size in self.last_var_snapshot)
    #     growth_ratio = total_growth_bytes / prev_total_bytes if prev_total_bytes > 0 else 0.0
        
    #     txt = f"\n        -> Growth = {total_growth_mb:.2f} MB"
    #     txt += f"\n        -> Ratio  = {growth_ratio * 100:.2f}%"
    #     txt += f"\n        -> Top {top_n} growing objects:"
        
    #     for oid, obj_repr, typ, diff in growth_list[:top_n]:
    #         txt += f"\n            id={oid}, type={typ}, repr={obj_repr[:50]} grew by {diff / (1024**2):.2f} MB"
        
    #     # Check thresholds
    #     if total_growth_mb > self.max_leak_mb or growth_ratio > self.max_leak_ratio:
    #         self.last_var_snapshot = snapshot
    #         raise RuntimeError(f"[ABORT] Python memory leak detected. Details:{txt}")
        
    #     # No leak -> update snapshot
    #     self.last_var_snapshot = snapshot
        
    #     return "    - Leak checked" + txt

    # def check_leak(self, top_n=5):
    #     gc.collect()
    #     top_objects = []

    #     for obj in gc.get_objects():
    #         try:
    #             size = sys.getsizeof(obj)
    #         except Exception:
    #             size = 0

    #         # If list not full, just append
    #         if len(top_objects) < top_n:
    #             top_objects.append((size, obj))
    #         else:
    #             # Replace the smallest if current is bigger
    #             min_index = min(range(len(top_objects)), key=lambda i: top_objects[i][0])
    #             if size > top_objects[min_index][0]:
    #                 top_objects[min_index] = (size, obj)

    #     # Sort descending by size
    #     top_objects.sort(key=lambda x: x[0], reverse=True)

    #     # Build report
    #     txt = f"Top {top_n} largest objects:\n"
    #     for size, obj in top_objects:
    #         try:
    #             repr_str = repr(obj)
    #         except Exception:
    #             repr_str = "<unrepresentable>"
    #         txt += f"  type={type(obj)}, size={size/(1024**3):.2f} MB, repr={repr_str[:50]}\n"

    #     return txt

    def make_log_entry(self, ram_result, cpu_result, gpu_result, mean_loop_time_result, watchdog_result, leak_result):
        """
        ## Log Runtime Check Results

        Writes the results of all enabled runtime checks to the configured log file.

        Logging only occurs if:
        - Logging is enabled
        - The current call index matches `log_every_x_calls`

        ### Parameters
        - **ram_result** (`str`): RAM check status
        - **cpu_result** (`str`): CPU check status
        - **gpu_result** (`str`): GPU check status
        - **mean_loop_time_result** (`str`): Mean loop time check status
        - **watchdog_result** (`str`): Watchdog timer status
        - **leak_result** (`str`): Memory leak check status
        """
        if self.should_log and (self.cur_call_n % self.log_every_x_calls == 0):
            now = datetime.datetime.now()
            log_txt = f"\n\n{'-'*32}\nDate: {now.day:02}.{now.month:02}.{now.year:04}\nTime: {now.hour:02}:{now.minute:02}\nLoops: {self.cur_call_n}"
            log_txt += f"\n{ram_result}"
            log_txt += f"\n{cpu_result}"
            log_txt += f"\n{gpu_result}"
            log_txt += f"\n{mean_loop_time_result}"
            log_txt += f"\n{watchdog_result}"
            log_txt += f"\n{leak_result}"
            log_txt += f"\n{'-'*32}"
            
            with open(self.log_path, "a") as file_:
                file_.write(log_txt)

    def make_abort_log(self, abort_message):
        """
        ## Log Abort Message

        Appends a critical abort message to the log file if logging is enabled.

        Used when:
        - Resource limits are exceeded
        - Watchdog triggers
        - Memory leaks are detected

        ### Parameters
        - **abort_message** (`str`): Abort message to write to the log
        """
        if self.should_log:
            with open(self.log_path, "a") as file_:
                file_.write("\n\n"+abort_message)

    def make_print_entry(self, ram_result, cpu_result, gpu_result, mean_loop_time_result, watchdog_result, leak_result):
        """
        ## Print Runtime Check Results

        Prints the results of all enabled runtime checks to the console.

        Printing only occurs if:
        - Printing is enabled
        - The current call index matches `print_every_x_calls`

        ### Parameters
        - **ram_result** (`str`): RAM check status
        - **cpu_result** (`str`): CPU check status
        - **gpu_result** (`str`): GPU check status
        - **mean_loop_time_result** (`str`): Mean loop time check status
        - **watchdog_result** (`str`): Watchdog timer status
        - **leak_result** (`str`): Memory leak check status
        """
        if self.should_print and (self.cur_call_n % self.print_every_x_calls == 0):
            now = datetime.datetime.now()
            print_txt = f"\nRunning Guard Check [{now.day:02}.{now.month:02}.{now.year:04} {now.hour:02}:{now.minute:02}]"
            print_txt += f"\n{ram_result}"
            print_txt += f"\n{cpu_result}"
            print_txt += f"\n{gpu_result}"
            print_txt += f"\n{mean_loop_time_result}"
            print_txt += f"\n{watchdog_result}"
            print_txt += f"\n{leak_result}"
            
            print(print_txt)

    def exit(self, msg=None):
        """
        ## Exit RuntimeGuard

        Terminates execution when a critical failure occurs.

        Behavior depends on configuration:
        - `hard_exit = True` → Immediate `os._exit(1)`
        - `hard_exit = False` → Raises `RuntimeError`

        ### Parameters
        - **msg** (`str | None`): Optional error message
        """
        if self.hard_exit:
            os._exit(1)
        else:
            raise RuntimeError(msg)
        
    def iteration_setting_check(self):
        """
        ## Validate Iteration Settings

        Ensures that:
        - `print_every_x_calls` is compatible with `update_every_x_calls`
        - `log_every_x_calls` is compatible with `update_every_x_calls`

        If misconfigured:
        - A warning is printed
        - The warning is logged (if logging is enabled)

        This prevents silent missing print or log entries.
        """
        # Check compatibility of update_every_x_calls with print/log intervals
        if self.should_print and self.print_every_x_calls % self.update_every_x_calls != 0:
            warning_msg = (
                f"[WARNING] print_every_x_calls ({self.print_every_x_calls}) "
                f"is not a multiple of update_every_x_calls ({self.update_every_x_calls}). "
                "Some print outputs may not occur."
            )
            print(warning_msg)
            if self.should_log:
                with open(self.log_path, "a") as f:
                    f.write("\n" + warning_msg)

        if self.should_log and self.log_every_x_calls % self.update_every_x_calls != 0:
            warning_msg = (
                f"[WARNING] log_every_x_calls ({self.log_every_x_calls}) "
                f"is not a multiple of update_every_x_calls ({self.update_every_x_calls}). "
                "Some log outputs may not occur."
            )
            print(warning_msg)
            with open(self.log_path, "a") as f:
                f.write("\n" + warning_msg)



# =====================
# || Context Classes ||
# =====================
class Run(object):
    """
    ## Run Context

    Wraps a loop iteration and automatically calls:
    - `start_loop()` on enter
    - `update()` on exit
    """
    def __init__(self, guard):
        self.guard = guard

    def __enter__(self):
        self.guard.start_loop()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.guard.update()



class Pause(object):
    """
    ## Pause Context

    Temporarily pauses watchdog monitoring inside a `with` block.
    """
    def __init__(self, guard):
        self.guard = guard

    def __enter__(self):
        self.guard.pause()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.guard.start_loop()




