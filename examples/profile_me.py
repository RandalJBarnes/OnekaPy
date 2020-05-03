import cProfile
import pstats
from pstats import SortKey

"""
o   How to line-by-line profile:
    - Add the decorator "@profile" to the functions of interest.
    - kernprof -l -v <driver>.py
    - python -m line_profiler <driver>.py.lprof > results.txt
"""


# -------------------------------------
def profile_me():
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()

    p = pstats.Stats(pr)
    p.strip_dirs()
    p.sort_stats(SortKey.TIME).print_stats(10)
