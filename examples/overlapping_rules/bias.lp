% var__(N) denotes a rule-variable slot, not a BK constant.
candidate_head(t(var__(1))).
allowed_literal(p(var__(1))).
allowed_literal(q(var__(1))).
allowed_literal(r(var__(1))).
max_body(2).

1 { head(H) : candidate_head(H) } 1.
0 { body(L) : allowed_literal(L) } 2.
#show head/1.
#show body/1.
