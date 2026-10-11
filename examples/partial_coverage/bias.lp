candidate_head(answer(var__(1))).
allowed_literal(p(var__(1))).
allowed_literal(q(var__(1))).
allowed_literal(r(var__(1))).
max_body(3).
1 { head(H) : candidate_head(H) } 1.
1 { body(L) : allowed_literal(L) } N :- max_body(N).
