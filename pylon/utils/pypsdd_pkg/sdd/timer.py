import sys
import time

class Timer:
    """Utility for timing code via Python's "with" statement.

    Examples
    --------

    >>> import time
    >>> with Timer("timing"):
    ...   time.sleep(2)
    ...
    = timing ... 2.002s
    >>> with Timer("timing",prefix="# "):
    ...   time.sleep(2)
    ...
    # timing ... 2.002s
    """

    def __init__(self,msg,prefix="= "):
        self.msg = msg
        self.prefix = prefix

    def __enter__(self):
        print(self.prefix + self.msg + " ...", end=' ')
        sys.stdout.flush()
        self.start = time.time()
        return self

    def __exit__(self,type,value,traceback):
        print("%.3fs" % (time.time()-self.start))
        sys.stdout.flush()
