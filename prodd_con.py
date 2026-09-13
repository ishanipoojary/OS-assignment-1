import threading
import queue
import time
import random

BUF_SIZE = 5
ITMS_TO_PRODC = 20

buf = queue.Queue(maxsize=BUF_SIZE)


def producer():
    for item in range(1, ITMS_TO_PRODC + 1):
        time.sleep(random.uniform(0.05, 0.20))
        buf.put(item)
        print(f"Producer has produced {item:02d} with size of buffer = {buf.qsize()}")

    buf.put(None)
    print("Producer has finished")


def consumer():
    while True:
        item = buf.get()

        if item is None:
            buf.task_done()
            print("Consumer has finished")
            break

        time.sleep(random.uniform(0.08, 0.25))
        print(f"Consumer has consumed {item:02d} and size of buffer = {buf.qsize()}")
        buf.task_done()


prod_thread = threading.Thread(target=producer)
cons_thread = threading.Thread(target=consumer)

prod_thread.start()
cons_thread.start()

prod_thread.join()
cons_thread.join()

print("\nProducer and Consumer executed.")