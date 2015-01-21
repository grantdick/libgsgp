#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <math.h>
#include <string.h>

#include <sys/time.h>

#include "mt19937ar.h"

#include "gsgp.h"

#include "data.h"
#include "readline.h"

struct data_set_details {
    bool scale_data;

    double **bounds;
    double  *centre;
    double  *scale;
    int n_features;

    double **train_X;
    double  *train_Y;
    int     *train_ids;
    int      n_train;
    double train_mean_rmse;

    double **test_X;
    double  *test_Y;
    int     *test_ids;
    int      n_test;
    double test_mean_rmse;
};

static double measure_rmse(struct gsgp_parameters *parameters, struct gsgp_individual *ind,
                    double **X, double *Y, int *id, int n,
                    double centre, double scale)
{
    int i;
    double y, yhat;
    double residual, mse;

    mse = 0;
    for (i = 0; i < n; ++i) {
        y    = scale * Y[i] + centre;
        yhat = scale * gsgp_exec_individual(ind, id[i], X[i], parameters) + centre;

        residual = (y - yhat) * (y - yhat);

        mse += (residual - mse) / (i + 1);
    }

    return sqrt(mse);
}

static double evaluate(struct gsgp_individual *ind, double **X, double *Y, int *ids, int n, struct gsgp_parameters *param, void *args)
{
    struct data_set_details *details;

    details = args;
    return measure_rmse(param, ind, X, Y, ids, n, details->centre[details->n_features], details->scale[details->n_features]);
}

static double mean_size(struct gsgp_individual **pop, int N)
{
    int i;
    double mean;

    mean = 0;
    for (i = 0; i < N; ++i) mean += (pop[i]->size - mean) / (i + 1);

    return mean;
}

static void print_generation(struct gsgp_individual **pop, int N, int g, int best,
                             struct gsgp_parameters *param, struct gsgp_archive *archive, void *args)
{
    struct data_set_details *details;
    double best_train, best_test;

    details = args;

    best_train = measure_rmse(param, pop[best], details->train_X, details->train_Y, details->train_ids, details->n_train, details->centre[details->n_features], details->scale[details->n_features]);
    best_test = measure_rmse(param, pop[best], details->test_X, details->test_Y, details->test_ids, details->n_test, details->centre[details->n_features], details->scale[details->n_features]);
    fprintf(stdout, "%4d %f %f %f %f %f %d\n", g,
            best_train, best_test,
            best_train / details->train_mean_rmse,
            best_test / details->test_mean_rmse,
            log(mean_size(pop, N)),
            gsgp_ensemble_size(pop[best], archive));
    fflush(stdout);
}

static void parse_param(char *line, char **key, char **value)
{
    *key = line;

    while (*line != '=') line++;
    *line = '\0';

    *value = line + 1;

    *key = trim(*key);
    *value = trim(*value);
}

static void parse_parameters(char *params_file, struct gsgp_parameters *params, struct data_set_details *details)
{
    FILE *input;
    char *buffer, *line, *key, *value;
    size_t bufsz;

    input = fopen(params_file, "r");
    buffer = line = NULL;
    bufsz = 0;

    line = next_line(&buffer, &bufsz, input);
    while (!feof(input)) {
        if (strlen(line) > 0) {
            parse_param(line, &key, &value);
            if (strncmp(key, "pop_size", 8) == 0) {
                params->N = atoi(value);
            } else if (strncmp(key, "generations", 11) == 0) {
                params->G = atoi(value);
            } else if (strncmp(key, "p_cross", 7) == 0) {
                params->p_cross = atof(value);
            } else if (strncmp(key, "p_mutation", 10) == 0) {
                params->p_mut = atof(value);
            } else if (strncmp(key, "tourn_size", 10) == 0) {
                params->k = atoi(value);
            } else if (strncmp(key, "min_depth", 9) == 0) {
                params->min_depth = atoi(value);
            } else if (strncmp(key, "max_depth", 9) == 0) {
                params->max_depth = atoi(value);
            } else if (strncmp(key, "ms", 2) == 0) {
                params->ms = atof(value);
            } else if (strncmp(key, "balanced_mutation", 17) == 0) {
                params->balanced_mutation = strncmp(value, "Y", 1) == 0;
            } else if (strncmp(key, "search_method", 13) == 0) {
                if (strncmp(value, "HILLCLIMB", 9) == 0) {
                    params->search_model = HILLCLIMB;
                } else if (strncmp(value, "POPULATION", 10) == 0) {
                    params->search_model = POPULATION_SEARCH;
                } else {
                    fprintf(stderr, "ERROR: Unknown value for parameter search_method: %s\n", value);
                    exit(EXIT_FAILURE);
                }
            } else if (strncmp(key, "model_type", 10) == 0) {
                if (strncmp(value, "STD_PARSE_TREE", 14) == 0) {
                    params->tree_model = STD_PARSE_TREE;
                } else if (strncmp(value, "NODIV_PARSE_TREE", 16) == 0) {
                    params->tree_model = NODIV_PARSE_TREE;
                } else if (strncmp(value, "INTERVAL_PARSE_TREE", 19) == 0) {
                    params->tree_model = INTERVAL_PARSE_TREE;
                } else if (strncmp(value, "REGRESSION_TREE", 14) == 0) {
                    params->tree_model = REGRESSION_TREE;
                } else {
                    fprintf(stderr, "ERROR: Unknown value for parameter model_type: %s\n", value);
                    exit(EXIT_FAILURE);
                }
            } else if (strncmp(key, "crossover_type", 14) == 0) {
                if (strncmp(value, "FIXED", 5) == 0) {
                    params->crossover_model = FIXED_PROPORTION;
                } else if (strncmp(value, "RANDOM", 6) == 0) {
                    params->crossover_model = RANDOM_PROPORTION;
                } else if (strncmp(value, "LEAST_SQUARES", 13) == 0) {
                    params->crossover_model = LEAST_SQUARES_CROSSOVER;
                } else {
                    fprintf(stderr, "ERROR: Unknown value for parameter crossover_type: %s\n", value);
                    exit(EXIT_FAILURE);
                }
            } else if (strncmp(key, "mutation_type", 13) == 0) {
                if (strncmp(value, "FIXED", 5) == 0) {
                    params->mutation_model = FIXED_SCALE;
                } else if (strncmp(value, "RANDOM", 6) == 0) {
                    params->mutation_model = RANDOM_SCALE;
                } else if (strncmp(value, "LEAST_SQUARES", 13) == 0) {
                    params->mutation_model = LEAST_SQUARES_MUTATION;
                } else {
                    fprintf(stderr, "ERROR: Unknown value for parameter mutation_type: %s\n", value);
                    exit(EXIT_FAILURE);
                }
            } else if (strncmp(key, "scale_data", 10) == 0) {
                details->scale_data = strncmp(value, "Y", 1) == 0;
            } else {
                fprintf(stderr, "WARNING: Unknown parameter: %s\n", key);
            }
        }
        line = next_line(&buffer, &bufsz, input);
    }
    fclose(input);
    free(buffer);
}

int main(int argc, char **argv)
{
    int i;

    struct gsgp_parameters *params;
    struct gsgp_archive    *archive;
    struct data_set_details details;

    struct timeval t;
    gettimeofday(&t, NULL);
    init_genrand(t.tv_usec);

    details.scale_data = false;

    params = gsgp_default_parameters();
    for (i = 4; i < argc; ++i) parse_parameters(argv[i], params, &details);

    load_fold(argv[1], argv[2], atoi(argv[3]),
              &(details.train_X), &(details.train_Y), &(details.n_train),
              &(details.test_X), &(details.test_Y), &(details.n_test),
              &(details.n_features),
              &(details.train_mean_rmse), &(details.test_mean_rmse),
              &(details.bounds),
              details.scale_data, &(details.centre), &(details.scale));

    details.train_ids = malloc(details.n_train * sizeof(int));
    details.test_ids = malloc(details.n_test * sizeof(int));

    for (i = 0; i < details.n_train; ++i) details.train_ids[i] = i;
    for (i = 0; i < details.n_test; ++i) details.test_ids[i] = i + details.n_train;

    gsgp_set_features(params, details.n_features, details.bounds);
    gsgp_set_rng(params, genrand_real2);

    archive = gsgp_create_archive(params->N*(params->G+1), details.n_train + details.n_test);

    gsgp_run(params, archive, details.train_X, details.train_Y, details.train_ids, details.n_train, evaluate, print_generation, &details);

    gsgp_release_archive(archive);
    gsgp_release_parameters(params);

    free(details.train_ids);
    free(details.test_ids);

    unload_data(details.train_X, details.train_Y, details.test_X, details.test_Y, details.bounds, details.centre, details.scale);

    return EXIT_SUCCESS;
}
