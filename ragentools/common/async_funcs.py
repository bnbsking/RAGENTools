import asyncio
from typing import Any, Callable, Dict, List


def async_executer(afunc: Callable, args_list: List[Dict]) -> Any:
    async def amain():
        tasks = []
        for args in args_list:
            tasks.append(afunc(**args))
        return await asyncio.gather(*tasks)
    return asyncio.run(amain())


def concurrency_wrapper(afunc: Callable, limit: int) -> Callable:
    semaphore = asyncio.Semaphore(limit)
    async def wrapped(**kwargs):
        async with semaphore:
            return await afunc(**kwargs)
    return wrapped


def batch_executer_for_func(inputs: List, batch_size: int, func: Callable) -> List:
    out = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        out.extend(func(batch))
    return out


async def batch_executer_for_afunc(inputs: List, batch_size: int, afunc: Callable) -> List:
    """Execute batch for afunc (intra-batch async). Not async at batch-level.
    asyncio.run cannot put inside async_func, use await instead.
    """
    out = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        out.extend(await afunc(batch))  # still wait for next batch
    return out
