import os, sys, absl
import threading, multiprocessing, traceback
import absl.logging as log
from functools import wraps
from timeit import default_timer
from typing import Callable

def timer(name:str) -> Callable:
    """Wrapper for calculating time costing of a function.

    Args:
        name (str): Identity for log.

    Raises:
        None

    Returns:
        decorator: A decorator.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*arg, **kwargs) -> None:
            start_time = default_timer()
            res = fn(*arg, **kwargs)
            elapse = default_timer()-start_time
            log.info("elapsed:{}:{}s".format(name, round(elapse, 3)))
            return res
        return wrapper

    return decorator

def new_thread(func:Callable) -> Callable:
    """Wrapper for opening a new thread to run function.

    Args:
        func (function): The function you need to run in another thread.

    Raises:
        None

    Returns:
        wrapper: A decorator.
    """

    def wrapper(*args):
        absl.logging.info(
            'Call `{}()` with process id: {}'.format(
                func.__name__,
                os.getpid()
            )
        )
        training_thread = threading.Thread(target=func, args=list(args))
        training_thread.start()

        return

    return wrapper

def new_process(func:Callable) -> Callable:
    """Wrapper for opening a new process to run function.

    Args:
        func (function): The function you need to run in another process.

    Raises:
        Exception

    Returns:
        wrapper: A decorator.
    """

    def target_func(queue, *args):
        try:
            absl.logging.info(
                'Call `{}()` with process id: {}'.format(
                    func.__name__,
                    os.getpid()
                )
            )
            result = func(*args)
            error = None
        except Exception:
            result = None
            ex_type, ex_value, tb = sys.exc_info()
            error = ex_type, ex_value,''.join(traceback.format_tb(tb))

        queue.put((result, error))

        return

    def wrapper(*args):
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=target_func, args=[queue] + list(args))
        p.start()
        result, error = queue.get()
        p.join()

        return result, error  

    return wrapper