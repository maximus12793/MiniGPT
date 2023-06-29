from collections import defaultdict

class PrefixSum:
    def __init__(self, s):
        # Initialize variables
        self.n = len(s)
        self.prefix_sums = defaultdict(lambda: [0] * (self.n + 1))

        # Compute prefix sums for each character
        for i in range(self.n):
            c = s[i]
            self.prefix_sums[c][i+1] = self.prefix_sums[c][i] + 1
        
        # Update prefix sums to represent total counts
        for c in self.prefix_sums:
            for i in range(1, self.n + 1):
                self.prefix_sums[c][i] += self.prefix_sums[c][i-1]

    def freq_count(self, c, start, end):
        # Compute frequency count using prefix sums
        return self.prefix_sums[c][end+1] - self.prefix_sums[c][start]


example = PrefixSum('capitalone')
print(example.prefix_sums)
print(example.freq_count('a', 0, 4))
print(example.freq_count('a', 0, 9))