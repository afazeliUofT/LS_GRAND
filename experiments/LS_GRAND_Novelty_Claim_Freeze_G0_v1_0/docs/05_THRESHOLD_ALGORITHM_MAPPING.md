# Mapping the original exact LS-GRAND theorem to the Threshold Algorithm

Let every latent state trajectory be one sorted list.  The objects are binary
candidate words.  A word's grade in list s is q_s(x;y), and the aggregate grade
is the sum over states.

The original exact LS-GRAND procedure:

1. performs sorted access to state queues;
2. when a word is encountered, obtains its complete sum score across states;
3. keeps the best complete valid word; and
4. uses the sum of current queue-head scores as an upper bound for every object
   not yet seen in any list.

This is the top-1 sum-aggregation Threshold Algorithm pattern.  A code-membership
predicate restricts the feasible objects, and communication-specific queue
construction remains technically meaningful, but the generic stopping theorem
must not be claimed as a new aggregation principle.

The later per-codeword partial-score refinement can be presented, at most, as a
communication-specific bound or implementation refinement.  Its novelty would
require a separate proof-level prior-art audit; it is not needed for the narrowed
practical project.
