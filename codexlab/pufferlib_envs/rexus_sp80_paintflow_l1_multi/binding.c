#include "rexus_sp80_paintflow_l1_multi.h"

#define OBS_SIZE 32
#define NUM_ATNS 1
#define ACT_SIZES {16}
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
    dict_set(out, "stack2_completion", log->stack2_completion);
    dict_set(out, "stack3_completion", log->stack3_completion);
    dict_set(out, "selective_edge_reroute", log->selective_edge_reroute);
    dict_set(out, "wrong_edge_activation", log->wrong_edge_activation);
    dict_set(out, "mean_required_objects", log->n > 0.0f ? log->required_objects_sum / log->n : 0.0f);
    dict_set(out, "mean_placed_objects", log->n > 0.0f ? log->placed_objects_sum / log->n : 0.0f);
}
