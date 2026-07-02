#include <math.h>
#include <stdlib.h>
#include "raylib.h"

#define GRID_W 14
#define GRID_H 15
#define MAX_STEPS 20
#define NUM_TARGETS 2
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
    float lateral_target_x_sum;
    float lateral_target_x_min;
    float lateral_target_x_max;
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
    int bar_x;
    int bar_y;
    int target;
    int target_x;
    int target_y;
    int tick;
    float episode_return;
} Env;

static const int TARGET_BASE_X[NUM_TARGETS] = {6, 6};
static const int TARGET_BASE_Y[NUM_TARGETS] = {6, 11};

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

static void compute_observations(Env* env) {
    int tx = env->target_x;
    int ty = env->target_y;
    float dx = (float)(tx - env->bar_x);
    float dy = (float)(ty - env->bar_y);
    float dist = abs_f(dx) + abs_f(dy);
    env->observations[0] = norm_x(env->bar_x);
    env->observations[1] = norm_y(env->bar_y);
    env->observations[2] = (float)env->target;
    env->observations[3] = norm_x(tx);
    env->observations[4] = norm_y(ty);
    env->observations[5] = dx / (float)GRID_W;
    env->observations[6] = dy / (float)GRID_H;
    env->observations[7] = dist / (float)(GRID_W + GRID_H);
    env->observations[8] = (float)env->tick / (float)MAX_STEPS;
    env->observations[9] = env->bar_x == tx ? 1.0f : 0.0f;
    env->observations[10] = env->bar_y == ty ? 1.0f : 0.0f;
    env->observations[11] = 1.0f;
}

static void add_episode_log(Env* env, float clean_catch, float overflow) {
    env->log.target_clean_catch += clean_catch;
    env->log.level_completion += clean_catch;
    env->log.overflow += overflow;
    env->log.perf += clean_catch;
    env->log.score += env->episode_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    env->log.lateral_target_x_sum += (float)env->target_x;
    if (env->log.n == 0.0f || (float)env->target_x < env->log.lateral_target_x_min) {
        env->log.lateral_target_x_min = (float)env->target_x;
    }
    if (env->log.n == 0.0f || (float)env->target_x > env->log.lateral_target_x_max) {
        env->log.lateral_target_x_max = (float)env->target_x;
    }
    env->log.n += 1.0f;
}

void c_reset(Env* env) {
    env->target = rand_r(&env->rng) % NUM_TARGETS;
    env->target_x = TARGET_LATERAL_MIN_X + (rand_r(&env->rng) % (TARGET_LATERAL_MAX_X - TARGET_LATERAL_MIN_X + 1));
    env->target_y = TARGET_BASE_Y[env->target];
    env->bar_x = rand_r(&env->rng) % GRID_W;
    env->bar_y = 3 + (rand_r(&env->rng) % 11);
    env->tick = 0;
    env->episode_return = 0.0f;
    compute_observations(env);
}

void c_step(Env* env) {
    env->tick += 1;
    int action = (int)env->actions[0];
    float reward = -0.01f;
    int tx = env->target_x;
    int ty = env->target_y;
    float before = abs_f((float)(tx - env->bar_x)) + abs_f((float)(ty - env->bar_y));

    if (action == 1) env->bar_y -= 1;
    else if (action == 2) env->bar_y += 1;
    else if (action == 3) env->bar_x -= 1;
    else if (action == 4) env->bar_x += 1;

    env->bar_x = clamp_i(env->bar_x, 0, GRID_W - 1);
    env->bar_y = clamp_i(env->bar_y, 0, GRID_H - 1);

    float after = abs_f((float)(tx - env->bar_x)) + abs_f((float)(ty - env->bar_y));
    reward += 0.04f * (before - after);

    if (action == 5) {
        float clean = env->bar_x == tx && env->bar_y == ty ? 1.0f : 0.0f;
        float near = env->bar_x == tx && abs_f((float)(env->bar_y - ty)) <= 1.0f ? 1.0f : 0.0f;
        if (clean > 0.0f) {
            reward += 10.0f;
        } else if (near > 0.0f) {
            reward += 2.0f;
        } else {
            reward -= 4.0f;
        }
        env->rewards[0] = reward;
        env->episode_return += reward;
        env->terminals[0] = 1.0f;
        add_episode_log(env, clean, clean > 0.0f ? 0.0f : 1.0f);
        c_reset(env);
        return;
    }

    if (env->tick >= MAX_STEPS) {
        reward -= 1.0f;
        env->rewards[0] = reward;
        env->episode_return += reward;
        env->terminals[0] = 1.0f;
        add_episode_log(env, 0.0f, 1.0f);
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
        InitWindow(GRID_W * cell, GRID_H * cell, "Rexus SP80 Paintflow");
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
    DrawRectangle(0, TARGET_BASE_Y[0] * cell, GRID_W * cell, cell, (Color){24, 64, 120, 255});
    DrawRectangle(0, TARGET_BASE_Y[1] * cell, GRID_W * cell, cell, (Color){20, 96, 58, 255});
    DrawRectangle(TARGET_BASE_X[0] * cell, TARGET_BASE_Y[0] * cell, cell, cell, (Color){48, 128, 255, 255});
    DrawRectangle(TARGET_BASE_X[1] * cell, TARGET_BASE_Y[1] * cell, cell, cell, (Color){40, 210, 120, 255});
    DrawRectangle(env->target_x * cell, env->target_y * cell, cell, cell, (Color){250, 96, 76, 255});
    DrawRectangle(env->bar_x * cell, env->bar_y * cell, cell * 2, cell / 2, (Color){245, 220, 85, 255});
    DrawCircle(env->target_x * cell + cell / 2, env->target_y * cell + cell / 2, 6, WHITE);
    EndDrawing();
}

void c_close(Env* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
