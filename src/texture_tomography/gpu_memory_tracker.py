class GPUMemoryTracker:
    def __init__(self):
        self.bytes = 0
        self.records = {}

    def add(self, name, arr):
        nbytes = arr.nbytes
        self.bytes += nbytes
        self.records[name] = self.records.get(name, 0) + nbytes

    def remove(self, name, arr):
        nbytes = arr.nbytes
        self.bytes -= nbytes
        self.records[name] -= nbytes
        if self.records[name] <= 0:
            del self.records[name]

    def summary(self):
        print("GPU memory usage:")
        for k, v in sorted(self.records.items()):
            print(f"  {k:30s}: {v/1024**2:8.2f} MB")
        print(f"TOTAL: {self.bytes/1024**2:.2f} MB")
