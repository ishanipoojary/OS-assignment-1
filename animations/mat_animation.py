import queue
import random
import threading
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

ROWS_A = 100
COLS_A = 100          
COLS_B = 100
NUM_WORKERS = 16
SEED_A = 101
SEED_B = 202

OUTPUT_GIF = "matrix_multiply_threaded.gif"
NUM_FRAMES = 140      
FPS = 4

TOTAL_TASKS = ROWS_A * COLS_B * COLS_A


def generate_matrix(rows, cols, min_val=1, max_val=9, seed=42):
    rng = random.Random(seed)
    return [[float(rng.randint(min_val, max_val)) for _ in range(cols)] for _ in range(rows)]


def task_at_index(idx, cols_b, cols_a):
    
    per_i = cols_b * cols_a
    i, rem = divmod(idx, per_i)
    j, k = divmod(rem, cols_a)
    return i, j, k


def run_instrumented_threaded_multiply(A, B, num_workers):
    
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B:
        raise ValueError("Dim of matrix mismatch for multiplication.")

    total_tasks = rows_A * cols_B * cols_A
    sample_every = max(1, total_tasks // NUM_FRAMES)

    C = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    cell_locks = [[threading.Lock() for _ in range(cols_B)] for _ in range(rows_A)]
    task_queue: queue.Queue = queue.Queue()

    instrumentation_lock = threading.Lock()  
    completed = 0
    worker_status = [("waiting", None, None, None) for _ in range(num_workers)]
    events = []

    def worker_loop(worker_id):
        nonlocal completed
        while True:
            task = task_queue.get()
            if task is None:
                with instrumentation_lock:
                    worker_status[worker_id] = ("stopped", None, None, None)
                task_queue.task_done()
                break

            i, j, k = task
            with instrumentation_lock:
                worker_status[worker_id] = ("processing", i, j, k)

           
            term = A[i][k] * B[k][j]
            with cell_locks[i][j]:
                C[i][j] += term
                new_value = C[i][j]
      

            with instrumentation_lock:
                completed += 1
                completed_now = completed
                if completed_now == 1 or completed_now == total_tasks or completed_now % sample_every == 0:
                    events.append(
                        {
                            "completed": completed_now,
                            "worker": worker_id,
                            "i": i, "j": j, "k": k,
                            "term": term,
                            "c_value": new_value,
                            "worker_status_snapshot": list(worker_status),
                            "queue_size": task_queue.qsize(),
                        }
                    )

            task_queue.task_done()

    threads = [
        threading.Thread(target=worker_loop, args=(w,), name=f"Worker-{w + 1}")
        for w in range(num_workers)
    ]
    for t in threads:
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

    events.sort(key=lambda e: e["completed"])
    return C, events


def sparse_verify(A, B, C, samples=8, seed=7):
   
    rng = random.Random(seed)
    rows, cols_b, inner = len(A), len(B[0]), len(A[0])
    max_diff = 0.0
    for _ in range(samples):
        i = rng.randrange(rows)
        j = rng.randrange(cols_b)
        expected = sum(A[i][k] * B[k][j] for k in range(inner))
        max_diff = max(max_diff, abs(expected - C[i][j]))
    return max_diff


def build_animation(A, B, events, total_tasks, num_workers):
    n = len(A)  # 100

    C_display = [[0.0 for _ in range(len(B[0]))] for _ in range(n)]
    c_vmax = ROWS_A * 9 * 9 * 0.55  # generous headroom above typical running sums

    fig = plt.figure(figsize=(15, 9.5))
    gs = GridSpec(2, 3, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.25,
                  top=0.88, bottom=0.06, left=0.04, right=0.98)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis("off")

    im_a = ax_a.imshow(A, cmap="Blues", vmin=1, vmax=9)
    im_b = ax_b.imshow(B, cmap="Purples", vmin=1, vmax=9)
    im_c = ax_c.imshow(C_display, cmap="OrRd", vmin=0, vmax=c_vmax)

    for ax, title in ((ax_a, "Matrix A (100x100)"),
                       (ax_b, "Matrix B (100x100)"),
                       (ax_c, "Matrix C (100x100, building up)")):
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im_a, ax=ax_a, shrink=0.75, pad=0.02)
    fig.colorbar(im_b, ax=ax_b, shrink=0.75, pad=0.02)
    cbar_c = fig.colorbar(im_c, ax=ax_c, shrink=0.75, pad=0.02)

    HL_COLOR = "#00e5ff"
    HL_SIZE = 3.4  # cells - drawn a bit larger than 1x1 so the highlight is easy to spot
    hl_a = Rectangle((0, 0), HL_SIZE, HL_SIZE, facecolor="none", edgecolor=HL_COLOR, linewidth=2.4)
    hl_b = Rectangle((0, 0), HL_SIZE, HL_SIZE, facecolor="none", edgecolor=HL_COLOR, linewidth=2.4)
    hl_c = Rectangle((0, 0), HL_SIZE, HL_SIZE, facecolor="none", edgecolor=HL_COLOR, linewidth=2.4)
    ax_a.add_patch(hl_a)
    ax_b.add_patch(hl_b)
    ax_c.add_patch(hl_c)

    label_a = ax_a.text(0, 0, "", color=HL_COLOR, fontsize=9, fontweight="bold",
                         ha="left", va="bottom")
    label_b = ax_b.text(0, 0, "", color=HL_COLOR, fontsize=9, fontweight="bold",
                         ha="left", va="bottom")
    label_c = ax_c.text(0, 0, "", color=HL_COLOR, fontsize=9, fontweight="bold",
                         ha="left", va="bottom")

    info_text = ax_info.text(
        0.0, 1.0, "", transform=ax_info.transAxes,
        family="monospace", fontsize=9.3, va="top", ha="left",
    )

    fig.suptitle(
        "100 x 100 Matrix Multiplication  |  16 Worker Threads  |  "
        "1,000,000 Scalar Multiplication Operations",
        fontsize=13, fontweight="bold",
    )

    half = HL_SIZE / 2

    def place_highlight(rect, label, row, col, value):
        rect.set_xy((col - half, row - half))
        offset = 4
        label.set_position((min(col + offset, n - 1), max(row - offset, 0)))
        label.set_text(f"[{row},{col}]={value:.0f}")

    def worker_lines(snapshot):
        lines = []
        for w, status in enumerate(snapshot):
            state, i, j, k = status
            if state == "processing":
                lines.append(f"Worker-{w + 1:02d}: processing A[{i:2d}][{k:2d}] x B[{k:2d}][{j:2d}]")
            elif state == "stopped":
                lines.append(f"Worker-{w + 1:02d}: finished, thread joined")
            else:
                lines.append(f"Worker-{w + 1:02d}: waiting for a task")
        mid = (len(lines) + 1) // 2
        left_col, right_col = lines[:mid], lines[mid:]
        merged = []
        for a_line, b_line in zip(left_col, right_col):
            merged.append(f"{a_line:<52}{b_line}")
        if len(left_col) > len(right_col):
            merged.append(left_col[-1])
        return merged

    def queue_preview(completed_count):
        upcoming = [task_at_index(idx, COLS_B, COLS_A)
                    for idx in range(completed_count, min(completed_count + 4, total_tasks))]
        return ", ".join(f"({i},{j},{k})" for i, j, k in upcoming) or "(queue drained)"

    def progress_bar(completed_count, width=40):
        frac = completed_count / total_tasks
        filled = int(frac * width)
        return f"[{'#' * filled}{'-' * (width - filled)}] {frac * 100:5.1f}%"

    def init():
        return [hl_a, hl_b, hl_c, label_a, label_b, label_c, info_text, im_c]

    def update(frame_idx):
        event = events[frame_idx]
        i, j, k, w = event["i"], event["j"], event["k"], event["worker"]
        term = event["term"]
        c_val = event["c_value"]
        completed_count = event["completed"]

        C_display[i][j] = c_val
        im_c.set_data(C_display)

        place_highlight(hl_a, label_a, i, k, A[i][k])
        place_highlight(hl_b, label_b, k, j, B[k][j])
        place_highlight(hl_c, label_c, i, j, c_val)

        worker_panel = "\n".join(worker_lines(event["worker_status_snapshot"]))

        trace = (
            f"CURRENT OPERATION (sample {frame_idx + 1}/{len(events)} of the real run)\n"
            f"  Worker-{w + 1:02d} dequeues task (i={i}, j={j}, k={k})\n"
            f"    term = A[{i}][{k}] * B[{k}][{j}]  =  {A[i][k]:.0f} * {B[k][j]:.0f}  =  {term:.0f}\n"
            f"    acquire cell_locks[{i}][{j}]  ->  C[{i}][{j}] += term  ->  new value = {c_val:.0f}\n"
            f"    release lock, task_queue.task_done(), worker requests next task"
        )

        queue_line = (
            f"Task queue (FIFO order, next ~4 of {total_tasks - completed_count:,} remaining): "
            f"{queue_preview(completed_count)}"
        )

        progress_line = (
            f"Operations processed: {completed_count:,} / {total_tasks:,}   "
            f"{progress_bar(completed_count)}"
        )

        info_text.set_text(
            f"{trace}\n\n"
            f"WORKER POOL (fixed pool of {num_workers} threads, repeatedly pulled from the shared queue)\n"
            f"{worker_panel}\n\n"
            f"{queue_line}\n"
            f"{progress_line}"
        )

        return [hl_a, hl_b, hl_c, label_a, label_b, label_c, info_text, im_c]

    anim = animation.FuncAnimation(
        fig, update, frames=len(events), init_func=init,
        interval=1000 / FPS, blit=False, repeat=True,
    )
    return fig, anim, cbar_c


def main():
    print(f"[1/4] Generating {ROWS_A}x{COLS_A} matrix A (seed {SEED_A}) and "
          f"{COLS_A}x{COLS_B} matrix B (seed {SEED_B})...")
    A = generate_matrix(ROWS_A, COLS_A, seed=SEED_A)
    B = generate_matrix(COLS_A, COLS_B, seed=SEED_B)

    print(f"[2/4] Running the real threaded multiply: {NUM_WORKERS} workers, "
          f"{TOTAL_TASKS:,} tasks...")
    t0 = time.perf_counter()
    C, events = run_instrumented_threaded_multiply(A, B, NUM_WORKERS)
    t1 = time.perf_counter()
    print(f"      Completed in {t1 - t0:.2f}s. Sampled {len(events)} operations "
          f"for the animation out of {TOTAL_TASKS:,} real ones.")

    max_diff = sparse_verify(A, B, C)
    print(f"      Spot-check vs direct computation: max diff = {max_diff:.2e}")

    print("[3/4] Building animation from the sampled operations...")
    fig, anim, _ = build_animation(A, B, events, TOTAL_TASKS, NUM_WORKERS)

    print(f"[4/4] Saving animation to {OUTPUT_GIF} (this can take a little while)...")
    anim.save(OUTPUT_GIF, writer=animation.PillowWriter(fps=FPS))
    plt.close(fig)

    print(f"Done. GIF saved as: {OUTPUT_GIF}")


if __name__ == "__main__":
    main()
