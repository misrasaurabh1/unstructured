from __future__ import annotations


def sorter(arr: list[int | float]) -> list[int | float]:
    print("codeflash stdout: Sorting list")
    arr.sort()
    print(f"result: {arr}")
    return arr
