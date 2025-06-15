#include <stddef.h>
#include "../core/cd_engine.h"

static int next_cyclic(CDState *st, int idx) {
    (void)st;
    return idx;
}

const CDIndexRule RULE_CYCLIC = {
    .init        = NULL,
    .begin_epoch = NULL,
    .next_j      = next_cyclic,
    .end_epoch   = NULL,
    .cleanup     = NULL
};
