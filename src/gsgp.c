#include <stdio.h>
#include <stdlib.h>

#include <stdbool.h>

#include <float.h>
#include <math.h>
#include <string.h>

#include "alloc.h"
#include "gsgp.h"

/***********************************************************************************************************************
 * EXTERNAL FUNCTIONS THAT WILL BE USED IN THIS MODULE AND DEFINED IN OTHER FILES
 **********************************************************************************************************************/
struct gsgp_tree *gsgp_build_tree(int min_depth, int max_depth,
                                  double **X, double *Y, int n,
                                  bool mutation,
                                  double interval[2],
                                  struct gsgp_parameters *param);

int gsgp_tree_size(struct gsgp_tree *tree);

void gsgp_release_tree(struct gsgp_tree *tree);

void gsgp_compute_interval(int op, double interval[2], double a, double b, double c, double d);

/***********************************************************************************************************************
 * INTERNAL FUNCTIONS THAT WILL BE USED IN THIS MODULE AND DEFINED LATER
 **********************************************************************************************************************/
static double default_rnd();

static void release_individual(struct gsgp_individual *ind);

static void add_to_archive(struct gsgp_individual *ind, struct gsgp_archive *archive, struct gsgp_parameters *param);

static void mark_references(struct gsgp_individual *ind);

static double perform_evaluation(struct gsgp_individual *ind, int id, double *features,
                                 struct gsgp_parameters *param);

static struct gsgp_individual *hillclimb(struct gsgp_parameters *param, struct gsgp_archive *archive,
                                         double **X, double *Y, int *ids, int n,
                                         double (*EVAL)(struct gsgp_individual *, double **, double *, int *, int
                                                        , struct gsgp_parameters *, void *),
                                         void   (*PRINT)(struct gsgp_individual **, int, int, int,
                                                         struct gsgp_parameters *, struct gsgp_archive *, void *),
                                         void   *opt);

static int select_parent(struct gsgp_individual **pop, int N, int k, double (*rnd)(void));

static struct gsgp_individual *search(struct gsgp_parameters *param, struct gsgp_archive *archive,
                                      double **X, double *Y, int *ids, int n,
                                      double (*EVAL)(struct gsgp_individual *, double **, double *, int *, int
                                                     , struct gsgp_parameters *, void *),
                                      void   (*PRINT)(struct gsgp_individual **, int, int, int,
                                                      struct gsgp_parameters *, struct gsgp_archive *, void *),
                                      void   *opt);









/***********************************************************************************************************************
 * BEGIN DEFININTION OF FUNCTIONS THAT DEFINE THE PUBLIC INTERFACE OF THIS MODULE
 **********************************************************************************************************************/
struct gsgp_parameters *gsgp_default_parameters()
{
    struct gsgp_parameters *def;

    def = ALLOC(1, sizeof(struct gsgp_parameters), false);

    def->search_model = POPULATION_SEARCH;
    def->tree_model = REGRESSION_TREE;
    def->crossover_coef = LEAST_SQUARES_CROSSOVER;
    def->mutation_coef = FIXED_SCALE;

    def->balanced_mutation = false;
    def->logistic_mutation = false;

    def->N = 200;
    def->G = 1000;

    def->p_cross = 0.3;
    def->p_mut = 0.6;

    def->tourn_k = 4;

    def->ms = 0.1;
    def->min_depth = 2;
    def->max_depth = 6;

    def->n_features = -1;
    def->feature_interval = NULL;

    def->rnd = default_rnd;

    return def;
}



void gsgp_release_parameters(struct gsgp_parameters *param)
{
    free(param->feature_interval[0]);
    free(param->feature_interval);
    free(param);
}



void gsgp_set_features(struct gsgp_parameters *param, int n_features, double **intervals)
{
    int i;

    if (param->feature_interval) {
        free(param->feature_interval[0]);
        free(param->feature_interval);
    }

    param->n_features = n_features;
    param->feature_interval = ALLOC(n_features, sizeof(double *), false);
    param->feature_interval[0] = ALLOC(2 * n_features, sizeof(double), false);
    for (i = 0; i < n_features; ++i) {
        param->feature_interval[i]    = param->feature_interval[0] + 2 * i;
        param->feature_interval[i][0] = (intervals == NULL) ? -INFINITY : intervals[i][0];
        param->feature_interval[i][1] = (intervals == NULL) ?  INFINITY : intervals[i][1];
    }
}



void gsgp_set_rng(struct gsgp_parameters *param, double (*rnd)(void))
{
    param->rnd = (rnd == NULL) ? default_rnd : rnd;
}



struct gsgp_archive *gsgp_create_archive(int initial_capacity, int n_instances)
{
    struct gsgp_archive *archive;

    archive = ALLOC(1, sizeof(struct gsgp_archive), false);

    archive->yhat_cache_size = n_instances;

    archive->next_model_id = 0;
    archive->N = 0;
    archive->capacity = initial_capacity;
    archive->instances = ALLOC(initial_capacity, sizeof(struct gsgp_individual *), true);

    return archive;
}



void gsgp_release_archive(struct gsgp_archive *archive)
{
    int i;

    for (i = 0; i < archive->N; ++i) release_individual(archive->instances[i]);

    free(archive->instances);
    free(archive);
}



void gsgp_prune_archive(struct gsgp_archive *archive, struct gsgp_individual **current, int N)
{
    int i, j;

    for (i = 0; i < archive->N; ++i) archive->instances[i]->referenced = false;

    for (i = 0; i < N; ++i) mark_references(current[i]);

    for (i = 0; i < archive->N; ++i) {
        if (archive->instances[i]->referenced) continue;

        do {
            release_individual(archive->instances[i]);
            archive->instances[i] = NULL;
            for (j = i + 1; j < archive->N; ++j) {
                archive->instances[j - 1] = archive->instances[j];
                archive->instances[j] = NULL;
            }

            archive->N--;
        } while ((archive->N > N) && archive->instances[i] && (!archive->instances[i]->referenced));
    }
}





struct gsgp_individual *gsgp_create_individual(int min_depth, int max_depth,
                                               double **X, double *Y, int n,
                                               struct gsgp_parameters *param,
                                               struct gsgp_archive *archive)
{
    struct gsgp_individual *ind;

    ind = ALLOC(1, sizeof(struct gsgp_individual), false);

    ind->type = BASE;

    ind->details.base.model_id = archive->next_model_id++;
    ind->details.base.model = gsgp_build_tree(min_depth, max_depth, X, Y, n, false, ind->interval, param);

    ind->size = gsgp_tree_size(ind->details.base.model);

    add_to_archive(ind, archive, param);

    return ind;
}



struct gsgp_individual *gsgp_crossover(struct gsgp_individual *mother, struct gsgp_individual *father,
                                       double **X, double *Y, int *ids, int n,
                                       struct gsgp_parameters *param, struct gsgp_archive *archive)
{
    struct gsgp_individual *ind;
    double ab[2], cd[2];

    int i;

    double y, m, f;
    double sum_mm, sum_mf, sum_my, sum_ff, sum_fy;

    ind = ALLOC(1, sizeof(struct gsgp_individual), false);

    ind->type = OFFSPRING;
    ind->details.parents.mother = mother;
    ind->details.parents.father = father;

    if (param->crossover_coef == LEAST_SQUARES_CROSSOVER || param->crossover_coef == MULTIPLE_LEAST_SQUARES_CROSSOVER) {
        /* we wish to find an "optimal" (with respect to the training
         * data) estimate of the crossover coefficients */
        sum_mm = sum_mf = sum_my = sum_ff = sum_fy = 0.0;
        for (i = 0; i < n; ++i) {
            y = Y[i];
            m = gsgp_exec_individual(mother, ids[i], X[i], param);
            f = gsgp_exec_individual(father, ids[i], X[i], param);

            sum_mm += m * m;
            sum_mf += m * f;
            sum_my += m * y;
            sum_ff += f * f;
            sum_fy += f * y;
        }
        if (param->crossover_coef == LEAST_SQUARES_CROSSOVER) {
            /* here, crossover is viewed as:
             *    offspring = p*mother + (1 - p)*father
             *              = father + p*(mother - father),
             * so the coefficient p can be determined through
             * least squares estimation to minimise the residuals */
            ind->details.parents.pm = (sum_my - sum_fy - sum_mf + sum_ff) / (sum_mm - 2*sum_mf + sum_ff);
            ind->details.parents.pf = 1 - ind->details.parents.pm;
        } else {
            /* here, crossover is viewed as:
             *    offspring = p*mother + q*father,
             * we now have two coefficients to determine, again through
             * least squares estimation to minimise the residuals */
            ind->details.parents.pf = (sum_fy * sum_mm - sum_my * sum_mf) / (sum_mm * sum_ff - sum_mf * sum_mf);
            ind->details.parents.pm = (sum_my - ind->details.parents.pf * sum_mf) / sum_mm;
        }
    } else if (param->crossover_coef == FIXED_PROPORTION){
        /* here, crossover is viewed as:
         *    offspring = 0.5*mother + 0.5*father,
         * so proceed accordingly */
        ind->details.parents.pf = ind->details.parents.pm = 0.5;
    } else {
        /* here, crossover is viewed as:
         *    offspring = p*mother + (1-p)*father,
         * so we proceed as per the "standard" GSGP crossover
         * routine */
        ind->details.parents.pm = COEF_LIMIT + (param->rnd() * (1 - 2 * COEF_LIMIT));
        ind->details.parents.pf = 1 - ind->details.parents.pm;
    }

    /* do some sanity checking - if the coefficients get way out of
     * hand, then we invalidate the crossover. This has the effect of
     * making the model's execution return NaN, so effectively rules
     * it out for selection as a parent in future iterations */
    if (!(fabs(ind->details.parents.pm) >= COEF_LIMIT && fabs(ind->details.parents.pf) >= COEF_LIMIT)) {
        ind->details.parents.pm = ind->details.parents.pf = NAN;
    }

    /* compute the interval of the offspring - this is done for
     * consistency, more than anything else (the intervals are only
     * used in tree initialisation, but it's always nice to know what
     * the intervals of execution will be, anyway) */
    gsgp_compute_interval(2, ab, mother->interval[0], mother->interval[1], ind->details.parents.pm, ind->details.parents.pm);
    gsgp_compute_interval(2, cd, father->interval[0], father->interval[1], ind->details.parents.pf, ind->details.parents.pf);
    gsgp_compute_interval(0, ind->interval, ab[0], ab[1], cd[0], cd[1]);

    /* in the traditional GSGP crossover, the resulting offspring tree
     * will be five nodes larger than the combined size of its parents
     * (one addition node, two multiplication nodes, and two constant
     * nodes for the coefficients) */
    ind->size = 5 + mother->size + father->size;

    add_to_archive(ind, archive, param);

    return ind;
}



struct gsgp_individual *gsgp_mutation(struct gsgp_individual *base,
                                      double **X, double *Y, int *ids, int n,
                                      struct gsgp_parameters *param, struct gsgp_archive *archive)
{
    struct gsgp_individual *mutant, *mut;
    double im[2];

    int i;
    double gx, sum_mr, sum_mm;
    double *residual;

    mutant = ALLOC(1, sizeof(struct gsgp_individual), false);
    mutant->type = MUTANT;
    mutant->details.mutation.base = base;
    mutant->details.mutation.ms   = param->ms;

    residual = ALLOC(n, sizeof(double), false);
    for (i = 0; i < n; ++i) residual[i] = Y[i] - gsgp_exec_individual(mutant->details.mutation.base, ids[i], X[i], param);

    mut = ALLOC(1, sizeof(struct gsgp_individual), false);
    mut->type = BASE;
    mut->details.base.model_id = archive->next_model_id++;
    mut->details.base.model = gsgp_build_tree(param->min_depth, param->max_depth, X, residual, n, true, mut->interval, param);
    mut->size = gsgp_tree_size(mut->details.base.model);
    add_to_archive(mut, archive, param);

    mutant->details.mutation.mut = mut;

    if (param->tree_model == REGRESSION_TREE) {
        /* as the regression tree for the mutant is built on the
         * residuals of the base model, its "best" estimate
         * coefficient will be one, so we may as well set that to be
         * the coefficient */
        mutant->details.mutation.ms = 1;
    } else if (param->mutation_coef == LEAST_SQUARES_MUTATION) {
        /* here, crossover is viewed as:
         *    mutant = base + ms*mutation
         * so the coefficient ms can be determined through
         * least squares estimation to minimise the residuals */
        sum_mr = sum_mm = 0;
        for (i = 0; i < n; ++i) {
            gx = gsgp_exec_individual(mut, ids[i], X[i], param);

            sum_mr += gx * residual[i];
            sum_mm += gx * gx;
        }

        mutant->details.mutation.ms = sum_mr / sum_mm;
    } else if (param->mutation_coef == RANDOM_SCALE) {
        mutant->details.mutation.ms = param->ms * param->rnd();
    } else {
        mutant->details.mutation.ms = param->ms;
    }

    /* do some sanity checking - if the coefficients get way out of
     * hand, then we invalidate the crossover. This has the effect of
     * making the model's execution return NaN, so effectively rules
     * it out for selection as a parent in future iterations */
    if (!(fabs(mutant->details.mutation.ms) >= COEF_LIMIT)) mutant->details.mutation.ms = NAN;

    /* compute the interval of the offspring - this is done for
     * consistency, more than anything else (the intervals are only
     * used in tree initialisation, but it's always nice to know what
     * the intervals of execution will be, anyway) */
    gsgp_compute_interval(2, im, mut->interval[0], mut->interval[1], mutant->details.mutation.ms, mutant->details.mutation.ms);
    gsgp_compute_interval(0, mutant->interval, base->interval[0], base->interval[1], im[0], im[1]);

    /* in the traditional GSGP mutation, the resulting offspring tree
     * will be three nodes larger than the combined size of its parents
     * (one addition node, one multiplication node, and one constant
     * node for the coefficient) */
    mutant->size = 3 + base->size + mut->size;

    add_to_archive(mutant, archive, param);

    free(residual);

    return mutant;
}



double gsgp_exec_individual(struct gsgp_individual *ind, int id, double *features, struct gsgp_parameters *param)
{
    return perform_evaluation(ind, id, features, param);
}





void   gsgp_print_individual(struct gsgp_individual *ind, FILE *out)
{
    switch(ind->type) {
    case BASE: gsgp_print_model(ind->details.base.model, out); break;
    case OFFSPRING:
        fprintf(out, "((%f * ", ind->details.parents.pm);
        gsgp_print_individual(ind->details.parents.mother, out);
        fprintf(out, ") + (%f * ", ind->details.parents.pf);
        gsgp_print_individual(ind->details.parents.father, out);
        fprintf(out, "))");
        break;
    case MUTANT:
        fprintf(out, "(");
        gsgp_print_individual(ind->details.mutation.base, out);
        fprintf(out, " + (%f * ", ind->details.mutation.ms);
        gsgp_print_individual(ind->details.mutation.mut, out);
        fprintf(out, "))");
        break;
    default: break;
    }
}



struct gsgp_individual *gsgp_run(struct gsgp_parameters *param, struct gsgp_archive *archive,
                                 double **X, double *Y, int *ids, int n,
                                 double (*EVAL)(struct gsgp_individual *, double **, double *, int *, int,
                                                struct gsgp_parameters *, void *),
                                 void   (*PRINT)(struct gsgp_individual **, int, int, int,
                                                 struct gsgp_parameters *, struct gsgp_archive *, void *),
                                 void   *opt)
{
    if (param->search_model == HILLCLIMB) {
        return hillclimb(param, archive, X, Y, ids, n, EVAL, PRINT, opt);
    } else {
        return search(param, archive, X, Y, ids, n, EVAL, PRINT, opt);
    }
}

/***********************************************************************************************************************
 * END OF PUBLIC INTERFACE
 **********************************************************************************************************************/










/***********************************************************************************************************************
 * BEGIN HELPER ROUTINES
 **********************************************************************************************************************/
static double default_rnd()
{
    return ((double)rand()) / ((double)RAND_MAX + 1.0);
}

static void release_individual(struct gsgp_individual *ind)
{
    if (ind->type == BASE) {
        gsgp_release_tree(ind->details.base.model);
        ind->details.base.model = NULL;
    }

    free(ind->cached);
    free(ind->yhat);
    free(ind);
}

static void mark_references(struct gsgp_individual *ind)
{
    if (ind->referenced) return; /* no need to visit twice */

    ind->referenced = true;

    if (ind->type == OFFSPRING) {
        mark_references(ind->details.parents.mother);
        mark_references(ind->details.parents.father);
    } else if (ind->type == MUTANT) {
        mark_references(ind->details.mutation.base);
        mark_references(ind->details.mutation.mut);
    }
}

static void add_to_archive(struct gsgp_individual *ind, struct gsgp_archive *archive, struct gsgp_parameters *param)
{
    int i;

    if (archive->N >= archive->capacity) {
        /* time to expand */
        do { archive->capacity *= 2; } while (archive->N >= archive->capacity);

        archive->instances = REALLOC(archive->instances, archive->capacity, sizeof(struct gsgp_individual *));

        for (i = archive->N; i < archive->capacity; ++i) archive->instances[i] = NULL;
    }

    ind->cached = ALLOC(archive->yhat_cache_size, sizeof(bool), false);
    ind->yhat   = ALLOC(archive->yhat_cache_size, sizeof(double), false);
    memset(ind->cached, false, archive->yhat_cache_size * sizeof(bool));

    archive->instances[archive->N++] = ind;
}


static double perform_evaluation(struct gsgp_individual *ind, int id, double *features,
                                 struct gsgp_parameters *param)
{
    double ret;

    if (id >= 0 && ind->cached[id]) {
        return ind->yhat[id];
    } else {
        switch (ind->type) {
        case BASE: default:
            ret = gsgp_execute_model(ind->details.base.model, features);
            break;
        case OFFSPRING:
            ret = ind->details.parents.pm * perform_evaluation(ind->details.parents.mother, id, features, param)
                + ind->details.parents.pf * perform_evaluation(ind->details.parents.father, id, features, param);
            break;
        case MUTANT:
            ret = perform_evaluation(ind->details.mutation.base, id, features, param)
                + ind->details.mutation.ms * perform_evaluation(ind->details.mutation.mut, id, features, param);
            break;
        }

        /* do some sanity checking on the output, keep it within manageable bounds */
        if (isnan(ret)) ret = BIG_LIMIT;
        if (isinf(ret)) ret = isinf(ret) * BIG_LIMIT;
        if (!(ret < BIG_LIMIT)) ret = BIG_LIMIT;
        if (!(ret > -BIG_LIMIT)) ret = -BIG_LIMIT;
        if (fabs(ret) < SMALL_LIMIT) ret = 0;

        if (id >= 0) {
            ind->cached[id] = true;
            ind->yhat[id] = ret;
        }

        return ret;
    }
}

static struct gsgp_individual *hillclimb(struct gsgp_parameters *param, struct gsgp_archive *archive,
                                         double **X, double *Y, int *ids, int n,
                                         double (*EVAL)(struct gsgp_individual *, double **, double *, int *, int
                                                        , struct gsgp_parameters *, void *),
                                         void   (*PRINT)(struct gsgp_individual **, int, int, int,
                                                         struct gsgp_parameters *, struct gsgp_archive *, void *),
                                         void   *opt)
{
    int i, g;
    struct gsgp_individual *base, *best, *mut;

    best = gsgp_create_individual(param->min_depth, param->max_depth, X, Y, n, param, archive);
    best->fitness = EVAL(best, X, Y, ids, n, param, opt);
    for (i = 1; i < param->N; ++i) {
        base = gsgp_create_individual(param->min_depth, param->max_depth, X, Y, n, param, archive);
        base->fitness = EVAL(base, X, Y, ids, n, param, opt);

        if (base->fitness < best->fitness) best = base;
    }

    base = best;
    gsgp_prune_archive(archive, &base, 1);
    PRINT(&base, 1, 0, 0, param, archive, opt);

    for (g = 1; g <= param->G; ++g) {
        best = base;
        for (i = 0; i < param->N; ++i) {
            mut = gsgp_mutation(base, X, Y, ids, n, param, archive);
            mut->fitness = EVAL(mut, X, Y, ids, n, param, opt);

            if (mut->fitness < best->fitness) best = mut;
        }

        base = best;
        gsgp_prune_archive(archive, &base, 1);
        PRINT(&base, 1, g, 0, param, archive, opt);
    }

    gsgp_prune_archive(archive, &base, 1);

    return base;
}

static int select_parent(struct gsgp_individual **pop, int N, int k, double (*rnd)(void)) {
    int i, c, pick;

    c = (int)(N * rnd());
    for (i = 1; i < k; ++i) {
        pick = (int)(N * rnd());
        if (pop[pick]->fitness < pop[c]->fitness) c = pick;
    }

    return c;
}

static struct gsgp_individual *search(struct gsgp_parameters *param, struct gsgp_archive *archive,
                                      double **X, double *Y, int *ids, int n,
                                      double (*EVAL)(struct gsgp_individual *, double **, double *, int *, int,
                                                     struct gsgp_parameters *, void *),
                                      void   (*PRINT)(struct gsgp_individual **, int, int, int,
                                                      struct gsgp_parameters *, struct gsgp_archive *, void *),
                                      void   *opt)
{
    int i, g;
    int best, mother, father;
    double p;
    int init_min_depth, init_max_depth;
    struct gsgp_individual **pop, **gen, **swp, *ret;

    pop = ALLOC(param->N, sizeof(struct gsgp_individual *), false);
    gen = ALLOC(param->N, sizeof(struct gsgp_individual *), false);

    best = 0;
    for (i = 0; i < param->N; ++i) {
        init_max_depth = param->min_depth + (i % (param->max_depth - param->min_depth + 1));
        init_min_depth = ((i % 2) == 0) ? init_max_depth : param->min_depth;
        pop[i] = gsgp_create_individual(init_min_depth, init_max_depth, X, Y, n, param, archive);

        pop[i]->fitness = EVAL(pop[i], X, Y, ids, n, param, opt);

        if (pop[i]->fitness < pop[best]->fitness) best = i;
    }
    PRINT(pop, param->N, 0, best, param, archive, opt);

    for (g = 1; g <= param->G; ++g) {
        gen[0] = pop[best]; /* elitism, keep the best of the previous gen */

        best = 0;
        for (i = 1; i < param->N; ++i) {
            mother = select_parent(pop, param->N, param->tourn_k, param->rnd);

            p = param->rnd();
            if (p < param->p_cross) {
                father = select_parent(pop, param->N, param->tourn_k, param->rnd);
                gen[i] = gsgp_crossover(pop[mother], pop[father], X, Y, ids, n, param, archive);
            } else if (p < (param->p_cross + param->p_mut)) {
                gen[i] = gsgp_mutation(pop[mother], X, Y, ids, n, param, archive);
            } else {
                gen[i] = pop[mother];
            }

            gen[i]->fitness = EVAL(gen[i], X, Y, ids, n, param, opt);
            if (gen[i]->fitness < gen[best]->fitness) best = i;
        }

        swp = pop;
        pop = gen;
        gen = swp;

        gsgp_prune_archive(archive, pop, param->N);

        PRINT(pop, param->N, g, best, param, archive, opt);
    }

    ret = pop[best];
    gsgp_prune_archive(archive, &ret, 1);

    free(pop);
    free(gen);

    return ret;
}
/***********************************************************************************************************************
 * END HELPER ROUTINES
 **********************************************************************************************************************/
