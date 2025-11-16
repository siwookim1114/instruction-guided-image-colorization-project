from Pipeline.pipe import Pipe
from collections import deque


class Pipeline:
    def __init__(self):
        self.pipes: list[Pipe] = []
        self.output_pipe = None

    def add_pipe(self, pipe: Pipe):
        self.pipes.append(pipe)
        return self

    def extend_pipe(self, pipes: list[Pipe]):
        self.pipes.extend(pipes)
        return self

    def set_output_pipe(self, pipe: Pipe):
        self.output_pipe = pipe

    def run(self):
        queue = deque(self.pipes)
        while queue:
            pipe = queue.pop()
            pipe.run()
            if not pipe.is_success():
                queue.appendleft(pipe)
        return self.output_pipe.result
