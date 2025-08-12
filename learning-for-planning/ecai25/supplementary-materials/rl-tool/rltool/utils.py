import random
import time

from collections import deque, OrderedDict


class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = deque(maxlen=max_memory)

    def add_trace(self, trace):
        self._samples.extend(trace)

    def add_sample(self, sample):
        self._samples.append(sample)

    def sample(self, n_samples):
        if n_samples > len(self._samples):
            return list(self._samples)
        else:
            return random.sample(self._samples, n_samples)

    def size(self):
        return len(self._samples)

    def empty(self):
        return len(self._samples) == 0

    def all(self):
        return list(self._samples)


class LRUDict(OrderedDict):
    '''An dict that can discard least-recently-used items, either by maximum capacity
    or by time to live.
    An item's ttl is refreshed (aka the item is considered "used") by direct access
    via [] or get() only, not via iterating over the whole collection with items()
    for example.
    Expired entries only get purged after insertions or changes. Either call purge()
    manually or check an item's ttl with ttl() if that's unacceptable.
    '''
    def __init__(self, *args, maxduration=None, maxsize=128, **kwargs):
        '''Same arguments as OrderedDict with these 2 additions:
        maxduration: number of seconds entries are kept. 0 or None means no timelimit.
        maxsize: maximum number of entries being kept.'''
        super().__init__(*args, **kwargs)
        self.maxduration = maxduration
        self.maxsize = maxsize
        self.purge()

    def purge(self):
        """Removes expired or overflowing entries."""
        if self.maxsize:
            # pop until maximum capacity is reached
            overflowing = max(0, len(self) - self.maxsize)
            for _ in range(overflowing):
                i = self.popitem(last=False)
        if self.maxduration:
            # expiration limit
            limit = time.time() - self.maxduration
            # as long as there are still items in the dictionary
            while self:
                # look at the oldest (front)
                _, lru = next(iter(super().values()))
                # if it is within the timelimit, we're fine
                if lru > limit:
                    break
                # otherwise continue to pop the front
                i = self.popitem(last=False)

    def __getitem__(self, key):
        # retrieve item
        value = super().__getitem__(key)[0]
        # update lru time
        super().__setitem__(key, (value, time.time()))
        self.move_to_end(key)
        return value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def ttl(self, key):
        '''Returns the number of seconds this item will live.
        The item might still be deleted if maxsize is reached.
        The time to live can be negative, as for expired items
        that have not been purged yet.'''
        if self.maxduration:
            lru = super().__getitem__(key)[1]
            return self.maxduration - (time.time() - lru)

    def __setitem__(self, key, value):
        super().__setitem__(key, (value, time.time()))
        self.purge()

    def items(self):
        # remove ttl from values
        return ((k, v) for k, (v, _) in super().items())

    def values(self):
        # remove ttl from values
        return (v for v, _ in super().values())


def get_lower_hard(histogram):
    iid = None
    m = max(int(1+max(histogram.values())), int(1.1*max(histogram.values())))
    t = 0
    for i in histogram.values():
        t += m-int(i)
    if t == 0:
        iid = random.randint(0, len(histogram)-1)
    else:
        r = random.randint(0, t-1)
        s = 0
        keys = list(histogram.keys())
        keys.sort()
        for i in keys:
            s += m - int(histogram[i])
            if r < s:
                iid = i
                break
    return iid


def get_lower_soft(histogram):
    iid = None
    solved = sum(histogram.values())
    r = random.randint(0, solved*(len(histogram)-1)-1)
    s = 0
    keys = list(histogram.keys())
    keys.sort()
    for i in keys:
        s += solved - histogram[i]
        if r < s:
            iid = i
            break
    return iid
