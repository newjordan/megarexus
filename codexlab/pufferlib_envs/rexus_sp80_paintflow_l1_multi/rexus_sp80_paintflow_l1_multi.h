#include <math.h>
#include <stdlib.h>
#include "raylib.h"

#define GRID_W 14
#define GRID_H 15
#define NUM_OBJECTS 3
#define MAX_STEPS 96
#define TARGET_LATERAL_MIN_X 1
#define TARGET_LATERAL_MAX_X (GRID_W - 2)

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float target_clean_catch;
    float overflow;
    float level_completion;
    float stack2_completion;
    float stack3_completion;
    float selective_edge_reroute;
    float wrong_edge_activation;
    float required_objects_sum;
    float placed_objects_sum;
    float n;
} Log;

typedef struct {
    Log log;
    int num_agents;
    unsigned int rng;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int obj_x[NUM_OBJECTS];
    int obj_y[NUM_OBJECTS];
    int target_x[NUM_OBJECTS];
    int target_y[NUM_OBJECTS];
    int required_count;
    int focus_edge;
    int tick;
    float episode_return;
} Env;

static const int TARGET_ROWS[NUM_OBJECTS] = {4, 8, 12};

static int clamp_i(int value, int lo, int hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

static float abs_f(float value) {
    return value < 0.0f ? -value : value;
}

static float norm_x(int x) {
    return (float)x / (float)(GRID_W - 1);
}

static float norm_y(int y) {
    return (float)y / (float)(GRID_H - 1);
}

static int object_placed(Env* env, int idx) {
    return env->obj_x[idx] == env->target_x[idx] && env->obj_y[idx] == env->target_y[idx];
}

static int placed_count(Env* env) {
    int count = 0;
    for (int i = 0; i < env->required_count; i++) {
        if (object_placed(env, i)) count += 1;
    }
    return count;
}

static float total_stack_distance(Env* env) {
    float total = 0.0f;
    for (int i = 0; i < env->required_count; i++) {
        total += abs_f((float)(env->target_x[i] - env->obj_x[i]));
        total += abs_f((float)(env->target_y[i] - env->obj_y[i]));
    }
    return total;
}

static void compute_observations(Env* env) {
    int idx = 0;
    env->observations[idx++] = (float)(env->required_count - 2);
    env->observations[idx++] = (float)env->focus_edge / (float)(NUM_OBJECTS - 1);
    env->observations[idx++] = (float)env->tick / (float)MAX_STEPS;
    env->observations[idx++] = (float)placed_count(env) / (float)env->required_count;
    for (int i = 0; i < NUM_OBJECTS; i++) {
        int active = i < env->required_count ? 1 : 0;
        float dx = (float)(env->target_x[i] - env->obj_x[i]);
        float dy = (float)(env->target_y[i] - env->obj_y[i]);
        env->observations[idx++] = (float)active;
        env->observations[idx++] = i == env->focus_edge ? 1.0f : 0.0f;
        env->observations[idx++] = norm_x(env->obj_x[i]);
        env->observations[idx++] = norm_y(env->obj_y[i]);
        env->observations[idx++] = norm_x(env->target_x[i]);
        env->observations[idx++] = norm_y(env->target_y[i]);
        env->observations[idx++] = dx / (float)GRID_W;
        env->observations[idx++] = dy / (float)GRID_H;
        env->observations[idx++] = object_placed(env, i) ? 1.0f : 0.0f;
    }
    env->observations[idx++] = total_stack_distance(env) / (float)(NUM_OBJECTS * (GRID_W + GRID_H));
}

static void add_episode_log(Env* env, float complete, float selective, float wrong_edge) {
    float overflow = complete > 0.0f ? 0.0f : 1.0f;
    env->log.target_clean_catch += complete;
    env->log.level_completion += complete;
    env->log.perf += complete;
    env->log.overflow += overflow;
    env->log.score += env->episode_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    env->log.stack2_completion += env->required_count == 2 ? complete : 0.0f;
    env->log.stack3_completion += env->required_count == 3 ? complete : 0.0f;
    env->log.selective_edge_reroute += selective;
    env->log.wrong_edge_activation += wrong_edge;
    env->log.required_objects_sum += (float)env->required_count;
    env->log.placed_objects_sum += (float)placed_count(env);
    env->log.n += 1.0f;
}

void c_reset(Env* env) {
    env->required_count = 2 + (rand_r(&env->rng) % 2);
    env->focus_edge = rand_r(&env->rng) % env->required_count;
    for (int i = 0; i < NUM_OBJECTS; i++) {
        env->target_x[i] = TARGET_LATERAL_MIN_X + (rand_r(&env->rng) % (TARGET_LATERAL_MAX_X - TARGET_LATERAL_MIN_X + 1));
        env->target_y[i] = TARGET_ROWS[i];
        env->obj_x[i] = rand_r(&env->rng) % GRID_W;
        env->obj_y[i] = 1 + (rand_r(&env->rng) % (GRID_H - 2));
    }
    env->tick = 0;
    env->episode_return = 0.0f;
    compute_observations(env);
}

void c_step(Env* env) {
    env->tick += 1;
    int action = (int)env->actions[0];
    float reward = -0.015f;
    float before = total_stack_distance(env);
    int activate_idx = -1;

    if (action >= 1 && action <= 12) {
        int obj = (action - 1) / 4;
        int move = (action - 1) % 4;
        if (obj < env->required_count) {
            if (move == 0) env->obj_y[obj] -= 1;
            else if (move == 1) env->obj_y[obj] += 1;
            else if (move == 2) env->obj_x[obj] -= 1;
            else if (move == 3) env->obj_x[obj] += 1;
            env->obj_x[obj] = clamp_i(env->obj_x[obj], 0, GRID_W - 1);
            env->obj_y[obj] = clamp_i(env->obj_y[obj], 0, GRID_H - 1);
        } else {
            reward -= 0.5f;
        }
    } else if (action >= 13 && action <= 15) {
        activate_idx = action - 13;
    }

    float after = total_stack_distance(env);
    reward += 0.05f * (before - after);
    reward += 0.02f * (float)placed_count(env);

    if (activate_idx >= 0) {
        int all_placed = placed_count(env) == env->required_count;
        int correct_edge = activate_idx == env->focus_edge;
        int active_edge = activate_idx < env->required_count;
        float complete = all_placed && correct_edge && active_edge ? 1.0f : 0.0f;
        float wrong_edge = active_edge && !correct_edge ? 1.0f : 0.0f;
        if (complete > 0.0f) {
            reward += 16.0f + 2.0f * (float)env->required_count;
        } else if (wrong_edge > 0.0f) {
            reward -= 8.0f;
        } else {
            reward -= 5.0f;
        }
        env->rewards[0] = reward;
        env->episode_return += reward;
        env->terminals[0] = 1.0f;
        add_episode_log(env, complete, complete, wrong_edge);
        c_reset(env);
        return;
    }

    if (env->tick >= MAX_STEPS) {
        reward -= 2.0f;
        env->rewards[0] = reward;
        env->episode_return += reward;
        env->terminals[0] = 1.0f;
        add_episode_log(env, 0.0f, 0.0f, 0.0f);
        c_reset(env);
        return;
    }

    env->rewards[0] = reward;
    env->episode_return += reward;
    compute_observations(env);
}

void c_render(Env* env) {
    const int cell = 32;
    if (!IsWindowReady()) {
        InitWindow(GRID_W * cell, GRID_H * cell, "Rexus SP80 Paintflow L1 Multi");
        SetTargetFPS(12);
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    BeginDrawing();
    ClearBackground((Color){8, 18, 24, 255});
    for (int y = 0; y < GRID_H; y++) {
        for (int x = 0; x < GRID_W; x++) {
            DrawRectangleLines(x * cell, y * cell, cell, cell, (Color){45, 60, 68, 255});
        }
    }
    Color rows[NUM_OBJECTS] = {
        (Color){52, 96, 180, 255},
        (Color){42, 150, 95, 255},
        (Color){150, 92, 180, 255},
    };
    Color objs[NUM_OBJECTS] = {
        (Color){250, 96, 76, 255},
        (Color){245, 220, 85, 255},
        (Color){110, 220, 245, 255},
    };
    for (int i = 0; i < env->required_count; i++) {
        DrawRectangle(0, TARGET_ROWS[i] * cell, GRID_W * cell, cell, rows[i]);
        DrawRectangle(env->target_x[i] * cell, env->target_y[i] * cell, cell, cell, (Color){245, 245, 245, 255});
        DrawCircle(env->target_x[i] * cell + cell / 2, env->target_y[i] * cell + cell / 2, 5, i == env->focus_edge ? RED : WHITE);
        DrawRectangle(env->obj_x[i] * cell, env->obj_y[i] * cell, cell, cell, objs[i]);
    }
    EndDrawing();
}

void c_close(Env* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
