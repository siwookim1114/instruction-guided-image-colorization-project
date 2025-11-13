from Pipeline.runner import Runner


class Pipe:
    unique_names = set()

    def __init__(self, runner: Runner, name):
        if name in self.unique_names:
            raise ValueError("Name must be unique")
        self.name = name
        self.unique_names.add(name)
        self.dependencies: list["Pipe"] = []
        self.runner = runner
        self.success = False
        self.result = None

    def __del__(self):
        self.unique_names.remove(self.name)

    def add_dependency(self, dependency: "Pipe"):
        self.dependencies.append(dependency)
        return self

    def run(self):
        if not self.ready():
            return

        self.result = self.runner.run({dp.name: dp.result for dp in self.dependencies})
        self.set_success()

    def set_runner(self, runner: Runner):
        self.runner = runner
        return self

    def set_success(self):
        self.success = True

    def is_success(self):
        return self.success

    def ready(self):
        return all([dep.is_success() for dep in self.dependencies])

    def __eq__(self, o: "Pipe"):
        return self.name == o.name
