#include <stdlib.h>
#include <time.h>
#include "../core/cd_engine.h"

static int rnd_init(CDState* st)
{
    (void)st;
    srand((unsigned)time(NULL));
    return 0;
}

static int rnd_next(CDState* st, int idx)
{
    (void)idx;
    return rand() % st->n;
}

const CDIndexRule RULE_RANDOM = {
    .init = rnd_init,
    .begin_epoch = NULL,
    .next_j = rnd_next,
    .end_epoch = NULL,
    .cleanup = NULL
};
