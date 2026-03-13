CREATE c1 capacity=16
CREATE c2 capacity=16
ANNOTATE c1 lower_to=channel
ANNOTATE c2 lower_to=channel
PUT c1 1
PUT c1 2
PUT c1 3
PUT c2 10
PUT c2 20
GET c1
GET c1
GET c2
