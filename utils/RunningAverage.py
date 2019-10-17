class RunningAverage:
    def __init__(self, max_samples=20):
        self.samples = []
        self.max_samples = max_samples
        self.running_average = 0

    def add_new_sample(self, new_sample):
        self.samples.append(new_sample)
        if len(self.samples) > self.max_samples:
            removed = self.samples.pop(0)
            self.running_average = (self.running_average * self.max_samples - removed + new_sample) / self.max_samples
        else:
            self.running_average = (self.running_average * (len(self.samples) - 1) + new_sample) / self.max_samples
        return self.running_average


