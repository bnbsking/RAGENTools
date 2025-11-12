from tenacity import retry, wait_fixed, stop_after_attempt
from typing import Any, Callable

def dynamic_retry(method: Callable) -> Callable:
    """Single-layer decorator that reads instance attributes."""
    def wrapper(self, *args, **kwargs) -> Any:  # <-- include *args and **kwargs
        # Create retry decorator dynamically using instance/class attributes
        decorator = retry(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_fixed(self.retry_sec)
        )
        return decorator(method)(self, *args, **kwargs)
    return wrapper


class MyClass:
    def __init__(self, retry_attempts=3, retry_sec=5):
        self.retry_sec = retry_sec
        self.retry_attempts = retry_attempts

    @dynamic_retry
    def my_method(self):
        print("Executing my_method")
        raise Exception("Simulated failure")


def f(x: int, y: int) -> int:
    return x + y

def wrapper(func: Callable) -> Callable:
    def wrap(y: int) -> int:
        return func(x=1, y=y)
    return wrap

import functools
h = functools.partial(f, x=1)

if __name__ == "__main__":
    g = wrapper(f)
    print(g(3))  # Should print 4
    print(h(y=3))  # Should also print 4
    
    obj = MyClass()
    obj.my_method()
    