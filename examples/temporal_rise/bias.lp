% Head slots: device=1, query time=2.
% Existential slots: current value=3, earlier value=4, earlier time=5.
candidate_head(rise(var__(1),var__(2))).
allowed_literal(reading(var__(1),var__(3),var__(2))).
allowed_literal(reading(var__(1),var__(4),var__(5))).
allowed_literal(increased(var__(4),var__(3))).
allowed_literal(before(var__(5),var__(2))).
max_body(4).

1 { head(H) : candidate_head(H) } 1.
0 { body(L) : allowed_literal(L) } 4.
#show head/1.
#show body/1.
