% Synthetic sensor readings for five devices.
reading(a,10,1). reading(a,16,2).
reading(b,20,1). reading(b,40,2).
reading(c,20,1). reading(c,21,2).
reading(d,40,1). reading(d,20,2).
reading(e,10,1). reading(e,16,3). reading(e,10,4).

value(V) :- reading(Device,V,T).
time(T) :- reading(Device,V,T).
increased(Previous,Current) :- value(Previous), value(Current),
    2 * Current >= 3 * Previous.
before(Earlier,Later) :- time(Earlier), time(Later), Earlier < Later.
