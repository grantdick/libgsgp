#ifndef GSGP
#define GSGP

#ifdef __cplusplus
extern "C" {
#endif

    #include <stdio.h>
    #include <stdbool.h>

    #define SMALL_LIMIT 1.0e-11
    #define BIG_LIMIT 1.0e15
    #define COEF_LIMIT SMALL_LIMIT

    struct gsgp_parameters {
        enum {
            HILLCLIMB,
            POPULATION_SEARCH
        } search_model;

        enum {
            STD_PARSE_TREE,
            NODIV_PARSE_TREE,
            INTERVAL_PARSE_TREE,
            REGRESSION_TREE
        } tree_model;

        enum {
            FIXED_PROPORTION,
            RANDOM_PROPORTION,
            LEAST_SQUARES_CROSSOVER
        } crossover_model;

        enum {
            FIXED_SCALE,
            RANDOM_SCALE,
            LEAST_SQUARES_MUTATION
        } mutation_model;

        bool balanced_mutation;

        int N;
        int G;

        double p_cross;
        double p_mut;

        int k;

        double ms;
        int min_depth;
        int max_depth;

        int n_features;

        double **feature_interval;

        double (*rnd)(void);
    };

    struct gsgp_tree; /* opaque structure for parse/regression tree details */

    struct gsgp_individual {
        enum { BASE, OFFSPRING, MUTANT } type;

        union {
            struct {
                int model_id;
                struct gsgp_tree *model;
            } base;

            struct {
                struct gsgp_individual *mother;
                struct gsgp_individual *father;
                double pm;
                double pf;
            } parents;

            struct {
                struct gsgp_individual *base;
                struct gsgp_individual *mut;
                double ms;
            } mutation;
        } details;

        bool   *cached;     /* an array that indicates whether or not
                             * a previous execution of a given data
                             * instance has taken place, if so, then
                             * the previous value will be returned,
                             * otherwise the individual is executed
                             * and the result is cached before
                             * returning */

        double *yhat;       /* the result of execution of the
                             * individual on a given data instance */

        double interval[2]; /* the lower and upper bounds of execution
                             * of this individual */

        double size;

        bool referenced;    /* true if individual is in the current
                             * population, or is an ancestor of an
                             * individual within the current
                             * population */

        double fitness;     /* the fitness assigned to the individual
                             * after evaluation (this could factor in
                             * scaling and other transformations */
    };

    struct gsgp_archive {
        int yhat_cache_size; /* the number of instances in our data
                              * set, each instance will be required to
                              * cache the result of execution for each
                              * instance that it encounters */

        int next_model_id;   /* the next identifier */
        int N;               /* the current size of the archive */
        int capacity;        /* the current maximum capacity of the archive */
        struct gsgp_individual **instances;
    };

    struct gsgp_parameters *gsgp_default_parameters();
    void gsgp_release_parameters(struct gsgp_parameters *param);
    void gsgp_set_features(struct gsgp_parameters *param, int n_features, double **intervals);
    void gsgp_set_rng(struct gsgp_parameters *param, double (*rnd)(void));

    struct gsgp_archive *gsgp_create_archive(int initial_capacity, int n_instances);
    void gsgp_prune_archive(struct gsgp_archive *archive, struct gsgp_individual **current, int N);
    void gsgp_release_archive(struct gsgp_archive *archive);

    struct gsgp_individual *gsgp_run(struct gsgp_parameters *param, struct gsgp_archive *archive,
                                     double **X, double *Y, int *idx, int n,
                                     double (*EVAL)(struct gsgp_individual *, double **, double *, int *, int,
                                                    struct gsgp_parameters *, void *),
                                     void   (*PRINT)(struct gsgp_individual **, int, int, int,
                                                     struct gsgp_parameters *, struct gsgp_archive *, void *),
                                     void   *opt);

    struct gsgp_individual *gsgp_create_individual(int min_depth, int max_depth,
                                                   double **X, double *Y, int n,
                                                   struct gsgp_parameters *param,
                                                   struct gsgp_archive *archive);
    struct gsgp_individual *gsgp_crossover(struct gsgp_individual *mother, struct gsgp_individual *father,
                                           double **X, double *Y, int *ids, int n,
                                           struct gsgp_parameters *param, struct gsgp_archive *archive);
    struct gsgp_individual *gsgp_mutation(struct gsgp_individual *base,
                                          double **X, double *Y, int *ids, int n,
                                          struct gsgp_parameters *param, struct gsgp_archive *archive);

    double gsgp_exec_individual(struct gsgp_individual *ind, int id, double *features, struct gsgp_parameters *param);

    void   gsgp_print_individual(struct gsgp_individual *ind, FILE *out);

    int    gsgp_ensemble_size(struct gsgp_individual *ind, struct gsgp_archive *archive);

    int    gsgp_extract_model(struct gsgp_individual *ind, double **beta_ptr, struct gsgp_tree ***models_ptr,
                              struct gsgp_archive *archive);

    void   gsgp_print_model(struct gsgp_tree *model, FILE *out);

    double gsgp_execute_model(struct gsgp_tree *model, double *data);

#ifdef __cplusplus
}
#endif

#endif
