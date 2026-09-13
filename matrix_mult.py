"""
================================================================================
Operating Systems Assignment - Task 2: Multithreaded Matrix Multiplication
================================================================================

Assignment Requirement:
-----------------------
"Implement matrix multiplication of two matrices using threads. Minimum 100 rows 
and 100 columns must be used. Every multiplication operation must be on a thread. 
The thread implementation must be demonstrated specifically in the calculation:
    sum[i][j] += A[i][k] * B[k][j]
Also demonstrate the working of the threaded matrix multiplication using an animation."

Technical Interpretation & OS Threading Architecture:
------------------------------------------------------
1. Matrix Dimension: 100 x 100 matrices (A and B), yielding a 100 x 100 result (C).
2. Operation Count: Standard matrix multiplication requires:
       Total scalar multiplications = 100 * 100 * 100 = 1,000,000 operations.
3. OS Thread Limit vs. Worker Thread Pool:
   - Creating 1,000,000 simultaneous native OS threads (`threading.Thread`) in Python
     on Windows results in OS thread stack memory allocation failure and 
     `RuntimeError: can't start new thread`.
   - To satisfy the requirement that "every multiplication operation must be on a thread"
     practically, we use a fixed pool of native worker threads (`threading.Thread`).
   - Every individual scalar multiplication task A[i][k] * B[k][j] is dispatched as a discrete
     work item to a thread worker.
   - The thread worker calculates `term = A[i][k] * B[k][j]` and executes:
         sum[i][j] += term
     under fine-grained cell locks to prevent race conditions during concurrent updates.
4. TensorRT Clarification:
   - TensorRT is NVIDIA's hardware-accelerated Deep Learning inference library for GPUs.
   - It is NOT applicable for CPU multithreading assignments in OS coursework.
   - This implementation relies entirely on Python's native `threading` library.
5. No NumPy rule:
   - Built-in matrix multiplication routines (e.g., np.matmul, np.dot) are NOT used.
   - All multiplications are computed manually cell-by-cell and term-by-term.
"""

import queue
import random
import sys
import threading
import time


def generate_matrix(rows: int, cols: int, min_val: int = 1, max_val: int = 10, seed: int = 42) -> list[list[float]]:
    """
    Programmatically generates a matrix of specified dimensions filled with random numbers.
    """
    rng = random.Random(seed)
    return [[float(rng.randint(min_val, max_val)) for _ in range(cols)] for _ in range(rows)]


def sequential_matrix_multiply(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """
    Sequential matrix multiplication baseline for correctness verification.
    Follows: sum[i][j] += A[i][k] * B[k][j]
    """
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    if cols_A != rows_B:
        raise ValueError("Matrix dimension mismatch for multiplication.")

    C = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]

    return C


def threaded_term_matrix_multiply(
    A: list[list[float]], B: list[list[float]], num_workers: int = 16
) -> list[list[float]]:
    """
    Multithreaded matrix multiplication executing term-level multiplication operations on threads.

    Each term task computes:
        term = A[i][k] * B[k][j]
    and accumulates:
        C[i][j] += term

    Race conditions are prevented using cell-level locks.
    All worker threads are explicitly waited on using join().
    """
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    if cols_A != rows_B:
        raise ValueError(f"Cannot multiply matrices: A columns ({cols_A}) != B rows ({rows_B})")

    # Initialize 100x100 result matrix with zeros
    C = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]

    # 2D array of fine-grained locks to prevent race conditions on C[i][j] accumulation
    cell_locks = [[threading.Lock() for _ in range(cols_B)] for _ in range(rows_A)]

    # Task queue with bounded maxsize to efficiently manage memory for 1,000,000 multiplication tasks
    task_queue: queue.Queue = queue.Queue(maxsize=20000)

    def worker_loop():
        """Worker thread entry point processing scalar multiplication tasks."""
        while True:
            task = task_queue.get()
            if task is None:
                task_queue.task_done()
                break

            i, j, k = task

            # --- Core Assignment Thread Calculation ---
            # 1. Perform individual multiplication operation on thread worker
            term = A[i][k] * B[k][j]

            # 2. Accumulate sum[i][j] += A[i][k] * B[k][j] under thread lock
            with cell_locks[i][j]:
                C[i][j] += term

            task_queue.task_done()

    # Spawn worker threads using Python threading module
    threads = []
    for w in range(num_workers):
        t = threading.Thread(target=worker_loop, name=f"MatrixWorkerThread-{w+1}")
        threads.append(t)
        t.start()

    # Enqueue all 1,000,000 individual scalar multiplication tasks (i, j, k)
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                task_queue.put((i, j, k))

    # Wait for all 1,000,000 multiplication tasks to complete
    task_queue.join()

    # Signal worker threads to terminate
    for _ in range(num_workers):
        task_queue.put(None)

    # Explicitly join all worker threads
    for t in threads:
        t.join()

    return C


def verify_correctness(C_threaded: list[list[float]], C_sequential: list[list[float]]) -> tuple[bool, float]:
    """
    Verifies that the threaded multiplication result matches the sequential version cell-by-cell.
    """
    rows, cols = len(C_threaded), len(C_threaded[0])
    max_diff = 0.0
    for i in range(rows):
        for j in range(cols):
            diff = abs(C_threaded[i][j] - C_sequential[i][j])
            if diff > max_diff:
                max_diff = diff
            if diff > 1e-4:
                return False, max_diff
    return True, max_diff


def main():
    print("=" * 80)
    print("  OPERATING SYSTEMS ASSIGNMENT - TASK 2: THREADED MATRIX MULTIPLICATION")
    print("=" * 80)
    print(f"Python Version: {sys.version.split()[0]} on {sys.platform}")
    print("Matrix Dimensions: 100 x 100 (Total 1,000,000 scalar multiplications)")
    print("-" * 80)

    # 1. Programmatically generate 100x100 matrices A and B
    print("\n[Step 1] Programmatically generating 100x100 matrices A and B...")
    rows, cols = 100, 100
    matrix_A = generate_matrix(rows, cols, min_val=1, max_val=10, seed=101)
    matrix_B = generate_matrix(cols, cols, min_val=1, max_val=10, seed=202)
    print(f"  - Matrix A created: {len(matrix_A)} rows x {len(matrix_A[0])} cols")
    print(f"  - Matrix B created: {len(matrix_B)} rows x {len(matrix_B[0])} cols")

    # 2. Run Threaded Matrix Multiplication (Term-Level Multithreading)
    num_workers = 16
    print(f"\n[Step 2] Executing Term-Level Threaded Multiplication ({num_workers} worker threads)...")
    print("  - Formula being evaluated on thread: sum[i][j] += A[i][k] * B[k][j]")
    print("  - Submitting 1,000,000 multiplication tasks to thread queue...")

    t0 = time.perf_counter()
    C_threaded = threaded_term_matrix_multiply(matrix_A, matrix_B, num_workers=num_workers)
    t1 = time.perf_counter()
    threaded_time = t1 - t0

    print(f"  [SUCCESS] Threaded multiplication completed in {threaded_time:.4f} seconds.")
    print(f"  - Joined all {num_workers} worker threads with join().")

    # 3. Run Sequential Baseline Matrix Multiplication for Verification
    print("\n[Step 3] Executing Sequential Reference Multiplication for Verification...")
    t2 = time.perf_counter()
    C_sequential = sequential_matrix_multiply(matrix_A, matrix_B)
    t3 = time.perf_counter()
    sequential_time = t3 - t2
    print(f"  - Sequential multiplication completed in {sequential_time:.4f} seconds.")

    # 4. Correctness Verification
    print("\n[Step 4] Verifying Threaded Result vs. Sequential Reference...")
    is_correct, max_difference = verify_correctness(C_threaded, C_sequential)

    if is_correct:
        print("  [OK] VERIFICATION PASSED: Threaded matrix result EXACTLY matches sequential result!")
        print(f"  - Maximum element difference: {max_difference:.6e}")
    else:
        print("  [FAIL] VERIFICATION FAILED: Discrepancy detected between threaded and sequential results!")
        print(f"  - Maximum difference: {max_difference:.6e}")

    # Display sample output cell calculations
    print("\n[Sample Result Check]")
    print(f"  - Cell C[0][0] Threaded  : {C_threaded[0][0]:.2f}")
    print(f"  - Cell C[0][0] Sequential: {C_sequential[0][0]:.2f}")
    print(f"  - Cell C[99][99] Threaded  : {C_threaded[99][99]:.2f}")
    print(f"  - Cell C[99][99] Sequential: {C_sequential[99][99]:.2f}")

    print("\n" + "=" * 80)
    print("  SUMMARY BENCHMARK")
    print("=" * 80)
    print(f"  Threaded Execution Time  : {threaded_time:.4f} s ({num_workers} threads)")
    print(f"  Sequential Execution Time: {sequential_time:.4f} s (1 thread)")
    print("=" * 80)


if __name__ == "__main__":
    main()
