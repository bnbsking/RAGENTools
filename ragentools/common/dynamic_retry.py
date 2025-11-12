from tenacity import retry, wait_fixed, stop_after_attempt
from typing import Any, Callable


def dynamic_retry(method: Callable) -> Callable:
    """
    This is a Decorator Wrapper,
    where decorator placed on method.
    When class initiate attributes `retry_attempts` and `retry_sec`,
    it will use those values for retrying the method.
    
    e.g. 
        class A:
            def __init__(self, retry_sec=5, retry_attempts=3):
                self.retry_sec = retry_sec
                self.retry_attempts = retry_attempts
            
            @dynamic_retry  # <-- apply decorator here
            def my_method(self):
                ...
    """
    retry_attempts: int = 3
    retry_sec: int = 5

    def log_exception(retry_state):
        print(f"Retrying {method.__name__} due to {retry_state.outcome.exception()}. "
              f"Attempt {retry_state.attempt_number} will start after {retry_state.next_action.sleep} seconds.")

    def wrapper(self, *args, **kwargs) -> Any:
        tries = self.retry_attempts if hasattr(self, 'retry_attempts') else retry_attempts
        waits = self.retry_sec if hasattr(self, 'retry_sec') else retry_sec
        decorator = retry(
            stop=stop_after_attempt(tries),
            wait=wait_fixed(waits),
            before_sleep=log_exception
        )
        return decorator(method)(self, *args, **kwargs)
    return wrapper


if __name__ == "__main__":
    class MyClass:
        def __init__(self, retry_attempts=3, retry_sec=5):
            self.retry_sec = retry_sec
            self.retry_attempts = retry_attempts

        @dynamic_retry
        def my_method(self):
            print("Executing my_method")
            raise Exception("Simulated failure")

    obj = MyClass()
    obj.my_method()