# Operating Systems Assignment

This repository contains the implementations for the Operating Systems assignment.

## Task 1: Producer-Consumer Problem

Implemented the Producer-Consumer problem using Python threads and a bounded queue.

- Buffer size: 5
- Total items: 20
- Uses Python `threading` and `queue`
- Demonstrates synchronization between the producer and consumer

### Run

    python prodd_con.py

## Task 2: Threaded Matrix Multiplication

Implemented matrix multiplication using multiple Python worker threads.

- Matrix size: 100 × 100
- 16 worker threads
- Each multiplication operation is processed by a worker thread
- Uses a thread-safe task queue
- Uses locks to safely update matrix elements
- Threaded result is verified against sequential matrix multiplication

### Run

    python matrix_mult.py

## Animation

An interactive HTML animation demonstrates the working of the threaded matrix multiplication, including the task queue, worker threads, multiplication operations, and locks.

    animations/matrix_multiplication.html
