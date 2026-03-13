CREATE c1 capacity=64
ANNOTATE c1 lower_to=objectfifo
PREFILL c1 0 1 2 3 4 5 6 7 8
PEEK c1 0
PEEK c1 1
PEEK c1 2
ADVANCE c1 1
