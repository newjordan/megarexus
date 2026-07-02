#include "rexus_sp80_paintflow.h"

#define OBS_SIZE 12
#define NUM_ATNS 1
#define ACT_SIZES {6}
#define OBS_TENSOR_T FloatTensor

#define Env Env
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "target_clean_catch", log->target_clean_catch);
    dict_set(out, "overflow", log->overflow);
    dict_set(out, "level_completion", log->level_completion);
    dict_set(out, "lateral_target_x_mean", log->n > 0.0f ? log->lateral_target_x_sum / log->n : 0.0f);
    dict_set(out, "lateral_target_x_min", log->lateral_target_x_min);
    dict_set(out, "lateral_target_x_max", log->lateral_target_x_max);
}
