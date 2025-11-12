import asyncio
import time
from typing import List

from ragentools.common.async_funcs import (
    async_executer,
    batch_executer_for_func,
    batch_executer_for_afunc,
    concurrency_wrapper
)

async def mock_task(id: int):
    await asyncio.sleep(1)
    return f"done-{id}"


def test_async_executer():
    # All tasks should complete in about 1 second
    args_list = [{"id": i} for i in range(10)]
    start = time.time()
    results = async_executer(mock_task, args_list)
    end = time.time()
    assert results == [f"done-{i}" for i in range(10)]
    assert end - start < 2
    print(f"Got results: {results} in {end - start:.2f} seconds")


def test_concurrency_wrapper():
    # With limit of 2 (concurrency), it should take about 5 seconds (5 batches) for 10 tasks
    limited_mock_task = concurrency_wrapper(mock_task, 2)
    args_list = [{"id": i} for i in range(10)]
    start = time.time()
    results = async_executer(limited_mock_task, args_list)
    end = time.time()
    assert results == [f"done-{i}" for i in range(10)]
    assert 4 <= end - start < 6
    print(f"Got results: {results} with concurrency limit 2 in {end - start:.2f} seconds")


def test_batch_executer_for_func():
    def process(data: List[int]) -> List[int]:
        return [x * 2 for x in data]
    result = batch_executer_for_func(inputs=list(range(6)), batch_size=3, func=process)
    assert result == [0, 2, 4, 6, 8, 10]
    print("batch_executer test passed:", result)


def test_batch_executer_for_afunc():
    async def process(data: List[int]) -> List[int]:
        await asyncio.sleep(1)
        return [x * 2 for x in data]
    result = asyncio.run(batch_executer_for_afunc(inputs=list(range(6)), batch_size=2, afunc=process))
    assert result == [0, 2, 4, 6, 8, 10]
    print("abatch_executer test passed:", result)


if __name__ == "__main__":
    # test_async_executer()
    # test_concurrency_wrapper()
    test_batch_executer_for_func()
    test_batch_executer_for_afunc()
