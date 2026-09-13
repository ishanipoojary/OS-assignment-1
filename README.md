# Operating Systems Assignment

This repository contains the implementations for the Operating Systems assignment.

## Task 1: Producer-Consumer Problem

Implemented the Producer-Consumer problem using Python threads and a bounded queue.

- Buffer size: 5
- Total items: 20
- Uses Python `threading` and `queue`
- Demonstrates synchronization between the producer and consumer

### Run

```bash
python prodd_con.py
```

## Task 2: Threaded Matrix Multiplication

Implemented matrix multiplication using multiple Python worker threads.

- Matrix size: 100 × 100
- 16 worker threads
- 1,000,000 individual multiplication operations
- Each multiplication operation is processed as a separate task by a worker thread
- Uses a thread-safe task queue
- Uses locks to safely update matrix elements
- Threaded result is verified against sequential matrix multiplication

### Run

```bash
python matrix_mult.py
```

## Animation

A Python-based animation demonstrates the working of the threaded matrix multiplication, including the matrices, worker threads, task queue, multiplication operations, locks, and computation progress.

The animation uses the same 100 × 100 matrix configuration and 16-worker-thread approach. Representative operations from the computation are visualized in the GIF.

### Generate Animation

```bash
python animations/mat_animation.py
```

The generated GIF is:

`matrix_multiply_threaded.gif`

The animation files are located in the `animations` folder.