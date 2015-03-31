#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <math.h>
#include <string.h>

#include "alloc.h"
#include "gsgp.h"

struct gsgp_tree {
    enum { FUNCTION, SPLIT, FEATURE, REGION } type;

    union {
        struct {
            struct gsgp_tree *arg0;
            struct gsgp_tree *arg1;
        } function;

        struct {
            int var;
            double point;

            struct gsgp_tree *left;
            struct gsgp_tree *right;
        } split;

        int feature;

        double region;
    } details;

    double (*exec)(struct gsgp_tree *, double *);
    void   (*print)(struct gsgp_tree *, FILE *);
};

static double min(double a, double b) { return (a < b) ? a : b; }
static double max(double a, double b) { return (a > b) ? a : b; }

static double logis(struct gsgp_tree *tree, double *data);
static double add(struct gsgp_tree *tree, double *data);
static double sub(struct gsgp_tree *tree, double *data);
static double mul(struct gsgp_tree *tree, double *data);
static double pdiv(struct gsgp_tree *tree, double *data);
static double feature(struct gsgp_tree *tree, double *data);

static void write_logis(struct gsgp_tree *tree, FILE *out);
static void write_add(struct gsgp_tree *tree, FILE *out);
static void write_sub(struct gsgp_tree *tree, FILE *out);
static void write_mul(struct gsgp_tree *tree, FILE *out);
static void write_pdiv(struct gsgp_tree *tree, FILE *out);
static void write_feature(struct gsgp_tree *tree, FILE *out);

static int select_operator(double a, double b, double c, double d, struct gsgp_parameters *param);

static struct gsgp_tree *build_parse_tree(int current_depth, int min_depth, int max_depth,
                                          double interval[2],
                                          struct gsgp_parameters *param);





static double split(struct gsgp_tree *tree, double *data);
static double region(struct gsgp_tree *tree, double *data);

static void write_split(struct gsgp_tree *tree, FILE *out);
static void write_region(struct gsgp_tree *tree, FILE *out);

static struct gsgp_tree *build_regression_tree(int current_depth, int min_depth, int max_depth,
                                               double **X, double *Y, int n,
                                               int *mask,
                                               double interval[2],
                                               struct gsgp_parameters *param);










void gsgp_compute_interval(int op, double interval[2], double a, double b, double c, double d)
{
    switch (op) {
    case 0: interval[0] = a + c; interval[1] = b + d; break;
    case 1: interval[0] = a - d; interval[1] = b - c; break;
    case 2:
        interval[0] = min(min(a*c, b*c), min(a*d, b*d));
        interval[1] = max(max(a*c, b*c), max(a*d, b*d));
        break;
    case 3:
        if (c < SMALL_LIMIT && d > -SMALL_LIMIT) {
            interval[0] = interval[1] = NAN;
        } else {
            gsgp_compute_interval(2, interval, a, b, 1/d, 1/c);
        }
        break;
    case 4:
        interval[0] = 1 / (1.0 + exp(-a));
        interval[1] = 1 / (1.0 + exp(-b));
        break;
    }
}

struct gsgp_tree *gsgp_build_tree(int min_depth, int max_depth,
                             double **X, double *Y, int n,
                             bool mutation,
                             double interval[2],
                             struct gsgp_parameters *param)
{
    int *mask;
    double ab[2], cd[2];
    struct gsgp_tree *res;

    if (mutation && param->balanced_mutation) {
        res = ALLOC(1, sizeof(struct gsgp_tree), false);
        res->type = FUNCTION;
        res->exec = sub;
        res->print = write_sub;
        if (param->logistic_mutation) {
            res->details.function.arg0 = ALLOC(1, sizeof(struct gsgp_tree), false);
            res->details.function.arg0->type = FUNCTION;
            res->details.function.arg0->exec = logis;
            res->details.function.arg0->print = write_logis;
            res->details.function.arg0->details.function.arg0 = gsgp_build_tree(min_depth - 1, max_depth - 1, X, Y, n, false, interval, param);
            res->details.function.arg0->details.function.arg1 = NULL;
            gsgp_compute_interval(4, ab, interval[0], interval[1], NAN, NAN);

            res->details.function.arg1 = ALLOC(1, sizeof(struct gsgp_tree), false);
            res->details.function.arg1->type = FUNCTION;
            res->details.function.arg1->exec = logis;
            res->details.function.arg1->print = write_logis;
            res->details.function.arg1->details.function.arg0 = gsgp_build_tree(min_depth - 1, max_depth - 1, X, Y, n, false, interval, param);
            res->details.function.arg1->details.function.arg1 = NULL;
            gsgp_compute_interval(4, cd, interval[0], interval[1], NAN, NAN);
        } else {
            res->details.function.arg0 = gsgp_build_tree(min_depth, max_depth, X, Y, n, false, ab, param);
            res->details.function.arg1 = gsgp_build_tree(min_depth, max_depth, X, Y, n, false, cd, param);
        }
        gsgp_compute_interval(1, interval, ab[0], ab[1], cd[0], cd[1]);
    } else if (mutation && param->logistic_mutation) {
        res = ALLOC(1, sizeof(struct gsgp_tree), false);
        res->type = FUNCTION;
        res->exec = logis;
        res->print = write_logis;
        res->details.function.arg0 = gsgp_build_tree(min_depth - 1, max_depth - 1, X, Y, n, false, interval, param);
        res->details.function.arg1 = NULL;
        gsgp_compute_interval(4, interval, interval[0], interval[1], NAN, NAN);
    } else if (param->tree_model == REGRESSION_TREE) {
        mask = ALLOC(n, sizeof(int), false);
        memset(mask, 1, n * sizeof(int));
        res = build_regression_tree(0, min_depth, max_depth, X, Y, n, mask, interval, param);
        free(mask);
    } else {
        res = build_parse_tree(0, min_depth, max_depth, interval, param);
    }

    return res;
}

int gsgp_tree_size(struct gsgp_tree *tree)
{
    if (tree == NULL) return 0;

    switch (tree->type) {
    case FUNCTION: return 1 + gsgp_tree_size(tree->details.function.arg0) + gsgp_tree_size(tree->details.function.arg1);
    case SPLIT:    return 1 + gsgp_tree_size(tree->details.split.left) + gsgp_tree_size(tree->details.split.right);
    case FEATURE:  return 1;
    case REGION:   default: return 1;
    }
}

void gsgp_release_tree(struct gsgp_tree *tree)
{
    if (tree == NULL) return;

    if (tree->type == FUNCTION) { gsgp_release_tree(tree->details.function.arg0); gsgp_release_tree(tree->details.function.arg1); }
    if (tree->type == SPLIT)    { gsgp_release_tree(tree->details.split.left); gsgp_release_tree(tree->details.split.right); }

    free(tree);
}

void gsgp_print_model(struct gsgp_tree *model, FILE *out)
{
    model->print(model, out);
}

double gsgp_execute_model(struct gsgp_tree *model, double *data)
{
    return model->exec(model, data);
}










static double split(struct gsgp_tree *tree, double *data)
{
    if (data[tree->details.split.var] <= tree->details.split.point) {
        return tree->details.split.left->exec(tree->details.split.left, data);
    } else {
        return tree->details.split.right->exec(tree->details.split.right, data);
    }
}

static double region(struct gsgp_tree *tree, double *data)
{
    return tree->details.region;
}

static void write_split(struct gsgp_tree *tree, FILE *out)
{
    fprintf(out, "ifelse(x%d < %f, ", tree->details.split.var, tree->details.split.point);
    tree->details.split.left->print(tree->details.split.left, out);
    fprintf(out, ", ");
    tree->details.split.right->print(tree->details.split.right, out);
    fprintf(out, ")");
}

static void write_region(struct gsgp_tree *tree, FILE *out)
{
    fprintf(out, "%f", tree->details.region);
}

static double masked_variance(double *Y, int *mask, int N, double *m)
{
    int i, cnt;
    double delta, mean, ssq;

    cnt = 0;
    mean = ssq = 0;

    for (i = 0; i < N; ++i) {
        if (mask[i] == 0) continue;

        delta = Y[i] - mean;
        mean += delta / ++cnt;
        ssq += delta * (Y[i] - mean);
    }

    if (m) *m = mean;

    if (cnt < 2) return 0;

    return ssq / (cnt - 1);
}

static struct gsgp_tree *build_regression_tree(int current_depth, int min_depth, int max_depth,
                                               double **X, double *Y, int n,
                                               int *mask,
                                               double interval[2],
                                               struct gsgp_parameters *param)
{
    struct gsgp_tree *tree;
    bool pick_function;

    int i, v, *tmask, *fmask;
    double rm, rv, p;
    double il[2], ir[2];

    tree  = ALLOC(1, sizeof(struct gsgp_tree), false);
    tmask = ALLOC(n, sizeof(int), false);
    fmask = ALLOC(n, sizeof(int), false);

    rv = masked_variance(Y, mask, n, &rm);

    if (rv < 1.0e-11) {
        pick_function = false;
    } else if (current_depth < min_depth) {
        pick_function = true;
    } else if (current_depth < max_depth) {
        pick_function = param->rnd() < 0.5;
    } else {
        pick_function = false;
    }

    if (pick_function) {
        v = param->rnd() * param->n_features;
        do { i = param->rnd() * n; } while (mask[i] == 0);
        p = X[i][v];

        for (i = 0; i < n; ++i) tmask[i] = mask[i] * ((X[i][v] <= p) ? 1 : 0);
        for (i = 0; i < n; ++i) fmask[i] = mask[i] * ((X[i][v] <= p) ? 0 : 1);

        for (pick_function = false, i = 0; (!pick_function) && (i < n); ++i) pick_function = fmask[i];
    }


    if (pick_function) {
        tree->type=SPLIT;

        tree->exec = split;
        tree->print = write_split;

        tree->details.split.var = v;
        tree->details.split.point = p;

        tree->details.split.left  = build_regression_tree(current_depth + 1, min_depth, max_depth, X, Y, n, tmask, il, param);
        tree->details.split.right = build_regression_tree(current_depth + 1, min_depth, max_depth, X, Y, n, fmask, ir, param);

        interval[0] = min(il[0], ir[0]);
        interval[1] = max(il[1], ir[1]);
    } else {
        tree->type = REGION;

        tree->exec = region;
        tree->print = write_region;

        tree->details.region = rm;
        interval[0] = interval[1] = rm;
    }

    free(tmask);
    free(fmask);

    return tree;
}










static double logis(struct gsgp_tree *tree, double *data)
{
    return 1.0 / (1.0 + exp(-(tree->details.function.arg0->exec(tree->details.function.arg0, data))));
}

static double add(struct gsgp_tree *tree, double *data)
{
    return tree->details.function.arg0->exec(tree->details.function.arg0, data) + tree->details.function.arg1->exec(tree->details.function.arg1, data);
}

static double sub(struct gsgp_tree *tree, double *data)
{
    return tree->details.function.arg0->exec(tree->details.function.arg0, data) - tree->details.function.arg1->exec(tree->details.function.arg1, data);
}

static double mul(struct gsgp_tree *tree, double *data)
{
    return tree->details.function.arg0->exec(tree->details.function.arg0, data) * tree->details.function.arg1->exec(tree->details.function.arg1, data);
}

static double pdiv(struct gsgp_tree *tree, double *data)
{
    double a;

    a = tree->details.function.arg1->exec(tree->details.function.arg1, data);

    return (fabs(a) <= SMALL_LIMIT) ? 1 : (tree->details.function.arg0->exec(tree->details.function.arg0, data) / a);
}

static double feature(struct gsgp_tree *tree, double *data)
{
    return data[tree->details.feature];
}

static void write_logis(struct gsgp_tree *tree, FILE *out)
{
    fprintf(out, "logis(");
    tree->details.function.arg0->print(tree->details.function.arg0, out);
    fprintf(out, ")");
}

static void write_add(struct gsgp_tree *tree, FILE *out)
{
    fprintf(out, "(");
    tree->details.function.arg0->print(tree->details.function.arg0, out);
    fprintf(out, " + ");
    tree->details.function.arg1->print(tree->details.function.arg1, out);
    fprintf(out, ")");
}

static void write_sub(struct gsgp_tree *tree, FILE *out)
{
    fprintf(out, "(");
    tree->details.function.arg0->print(tree->details.function.arg0, out);
    fprintf(out, " - ");
    tree->details.function.arg1->print(tree->details.function.arg1, out);
    fprintf(out, ")");
}

static void write_mul(struct gsgp_tree *tree, FILE *out)
{
    fprintf(out, "(");
    tree->details.function.arg0->print(tree->details.function.arg0, out);
    fprintf(out, " * ");
    tree->details.function.arg1->print(tree->details.function.arg1, out);
    fprintf(out, ")");
}

static void write_pdiv(struct gsgp_tree *tree, FILE *out)
{
    fprintf(out, "pdiv(");
    tree->details.function.arg0->print(tree->details.function.arg0, out);
    fprintf(out, ", ");
    tree->details.function.arg1->print(tree->details.function.arg1, out);
    fprintf(out, ")");
}

static void write_feature(struct gsgp_tree *tree, FILE *out)
{
    fprintf(out, "x%d", tree->details.feature + 1);
}

static int select_operator(double a, double b, double c, double d, struct gsgp_parameters *param)
{
    if (param->tree_model == NODIV_PARSE_TREE) return 3 * param->rnd();
    if (param->tree_model == INTERVAL_PARSE_TREE && c < SMALL_LIMIT && d > -SMALL_LIMIT) return 3 * param->rnd();
    return 4 * param->rnd();
}

static struct gsgp_tree *build_parse_tree(int current_depth, int min_depth, int max_depth,
                                          double interval[2],
                                          struct gsgp_parameters *param)
{
    int op;
    struct gsgp_tree *tree;
    bool pick_function;
    double ab[2], cd[2];

    tree = ALLOC(1, sizeof(struct gsgp_tree), false);

    if (current_depth < min_depth) {
        pick_function = true;
    } else if (current_depth == max_depth) {
        pick_function = false;
    } else if (param->rnd() < (param->n_features / (double)((param->tree_model == NODIV_PARSE_TREE) ? 3 : 4) + param->n_features)) {
        pick_function = false;
    } else {
        pick_function = true;
    }

    if (pick_function) {
        tree->type = FUNCTION;

        tree->details.function.arg0 = build_parse_tree(current_depth + 1, min_depth, max_depth, ab, param);
        tree->details.function.arg1 = build_parse_tree(current_depth + 1, min_depth, max_depth, cd, param);

        op = select_operator(ab[0], ab[1], cd[0], cd[1], param);

        switch (op) {
        case 0:
            tree->exec = add;
            tree->print = write_add;
            break;
        case 1:
            tree->exec = sub;
            tree->print = write_sub;
            break;
        case 2:
            tree->exec = mul;
            tree->print = write_mul;
            break;
        case 3: default:
            tree->exec = pdiv;
            tree->print = write_pdiv;
            break;
        }

        gsgp_compute_interval(op, interval, ab[0], ab[1], cd[0], cd[1]);
    } else {
        tree->type = FEATURE;

        tree->details.feature = param->n_features * param->rnd();
        tree->exec = feature;
        tree->print = write_feature;

        interval[0] = param->feature_interval[tree->details.feature][0];
        interval[1] = param->feature_interval[tree->details.feature][1];
    }

    return tree;
}
