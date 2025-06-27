from typing import List, Union


def sorter(arr: Union[List[int], List[float]]) -> Union[List[int], List[float]]:
    # Use Python's built-in sort (Timsort, O(n log n)) for much better performance
    arr.sort()
    return arr
