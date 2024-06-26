import queue

q = queue.PriorityQueue()
q.put((1, 'a'))
print(q.get(block=False))
print(q.get(block=False))
