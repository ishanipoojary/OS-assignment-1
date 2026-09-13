import queue
import random
import sys
import threading
import time


def generate_matrix(rows: int, cols: int, min_val: int = 1, max_val: int = 10, seed: int = 42) -> list[list[float]]:
    rng = random.Random(seed)
    return [[float(rng.randint(min_val, max_val)) for _ in range(cols)] for _ in range(rows)]


def sequential_matrix_multiply(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    if cols_A != rows_B:
        raise ValueError("Dim of matrix mismatchs for multiplication.")

    C = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]

    return C


def threaded_term_matrix_multiply(
    A: list[list[float]], B: list[list[float]], num_workers: int = 16
) -> list[list[float]]:
    
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    if cols_A != rows_B:
        raise ValueError(f"Cannot multiply matrices: A columns ({cols_A}) != B rows ({rows_B})")

    C = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]

    cell_locks = [[threading.Lock() for _ in range(cols_B)] for _ in range(rows_A)]

    task_queue: queue.Queue = queue.Queue(maxsize=20000)

    def worker_loop():
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

    threads = []
    for w in range(num_workers):
        t = threading.Thread(target=worker_loop, name=f"MatrixWorkerThread-{w+1}")
        threads.append(t)
        t.start()

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                task_queue.put((i, j, k))

    task_queue.join()

    for _ in range(num_workers):
        task_queue.put(None)

    for t in threads:
        t.join()

    return C


def verify_correctness(C_threaded: list[list[float]], C_sequential: list[list[float]]) -> tuple[bool, float]:
    
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

    print("\n[Step 1] Programmatically generating 100x100 matrices A and B.....")
    rows, cols = 100, 100
    matrix_A = generate_matrix(rows, cols, min_val=1, max_val=10, seed=101)
    matrix_B = generate_matrix(cols, cols, min_val=1, max_val=10, seed=202)
    print(f"  - Matrix A created: {len(matrix_A)} rows x {len(matrix_A[0])} cols")
    print(f"  - Matrix B created: {len(matrix_B)} rows x {len(matrix_B[0])} cols")

    num_workers = 16
    print(f"\n[Step 2] Executing Term-Level Threaded Multiplication ({num_workers} worker threads)...")
    print("  - Formula being evaluated on thread: sum[i][j] += A[i][k] * B[k][j]")
    print("  - Submitting 1,000,000 multiplication tasks to thread queue...")

    t0 = time.perf_counter()
    C_threaded = threaded_term_matrix_multiply(matrix_A, matrix_B, num_workers=num_workers)
    t1 = time.perf_counter()
    threaded_time = t1 - t0

    print(f"  [SUCCESS] Threaded multiplication completed in {threaded_time:.4f} sec...")
    print(f"  - Joined all {num_workers} worker threads with join().")

    print("\n[Step 3] Executing Sequential Reference Multiplication for Verification......")
    t2 = time.perf_counter()
    C_sequential = sequential_matrix_multiply(matrix_A, matrix_B)
    t3 = time.perf_counter()
    sequential_time = t3 - t2
    print(f"  - Sequential multiplication completed in {sequential_time:.4f} sec.")

    print("\n[Step 4] Verifying Threaded Result vs Sequential Reference.........")
    is_correct, max_difference = verify_correctness(C_threaded, C_sequential)

    if is_correct:
        print("  [OK] VERIFICATION PASSED: Threaded matrix result matches sequential result exactly")
        print(f"  - Maximum element difference: {max_difference:.6e}")
    else:
        print("  [FAIL] VERIFICATION FAILED: Diff between threaded and sequential results")
        print(f"  - Maximum difference: {max_difference:.6e}")

    print("\n[Sample Result Check]")
    print(f"  - Cell C[0][0] Threaded  : {C_threaded[0][0]:.2f}")
    print(f"  - Cell C[0][0] Sequential: {C_sequential[0][0]:.2f}")
    print(f"  - Cell C[99][99] Threaded  : {C_threaded[99][99]:.2f}")
    print(f"  - Cell C[99][99] Sequential: {C_sequential[99][99]:.2f}")

    print("\n" + "=" * 80)
    print("  SUMMARY BENCHMARK")
    print("=" * 80)
    print(f"  Threaded Exectn Time  : {threaded_time:.4f} s ({num_workers} threads)")
    print(f"  Sequential Exectn Time: {sequential_time:.4f} s (1 thread)")
    print("=" * 80)


if __name__ == "__main__":
    main()
