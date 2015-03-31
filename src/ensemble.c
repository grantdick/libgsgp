#include <stdio.h>
#include <stdlib.h>

#include <stdbool.h>

#include "alloc.h"
#include "gsgp.h"

/***********************************************************************************************************************
 * INTERNAL FUNCTIONS THAT WILL BE USED IN THIS MODULE AND DEFINED LATER
 **********************************************************************************************************************/
static void mark_references(struct gsgp_individual *ind);

static int extract_models(struct gsgp_individual *ind,
                               struct gsgp_tree ***model_buffer, int N, int *buffer_capacity);
static double *extract_coef(struct gsgp_individual *ind, struct gsgp_tree **model_buffer, int N);
static int add_model(struct gsgp_tree *model, struct gsgp_tree ***model_buffer, int N, int *buffer_capacity);










/***********************************************************************************************************************
 * BEGIN DEFININTION OF FUNCTIONS THAT DEFINE THE PUBLIC INTERFACE OF THIS MODULE
 **********************************************************************************************************************/
int    gsgp_ensemble_size(struct gsgp_individual *ind, struct gsgp_archive *archive)
{
    int i, N;

    for (i = 0; i < archive->N; ++i) archive->instances[i]->referenced = false;

    mark_references(ind);

    N = 0;
    for (i = 0; i < archive->N; ++i) if (archive->instances[i]->type == BASE && archive->instances[i]->referenced) N++;

    return N;
}



int gsgp_extract_model(struct gsgp_individual *ind, double **beta_ptr, struct gsgp_tree ***models_ptr,
                       struct gsgp_archive *archive)
{
    int N, buffer_capacity;
    struct gsgp_tree **model_buffer;

    /* first, identify the base models used by the individual */
    buffer_capacity = 10000; /* I sincerely hope that it never gets bigger! */
    model_buffer = ALLOC(buffer_capacity, sizeof(struct gsgp_tree *), false);
    N = extract_models(ind, &model_buffer, 0, &buffer_capacity);

    if (beta_ptr) *beta_ptr = extract_coef(ind, model_buffer, N);

    if (models_ptr) {
        *models_ptr = REALLOC(model_buffer, N, sizeof(struct gsgp_tree *));
    } else {
        free(model_buffer);
    }

    return N;
}

/***********************************************************************************************************************
 * END OF PUBLIC INTERFACE
 **********************************************************************************************************************/










/***********************************************************************************************************************
 * BEGIN HELPER ROUTINES
 **********************************************************************************************************************/
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

static int add_model(struct gsgp_tree *model, struct gsgp_tree ***model_buffer, int N, int *buffer_capacity)
{
    int i;

    /* first, check that the model is not already present */
    for (i = 0; i < N; ++i) if (model == (*model_buffer)[i]) return N;

    if (N >= *buffer_capacity) {
        do { *buffer_capacity = 2 * (*buffer_capacity); } while (N >= *buffer_capacity);
        *model_buffer = REALLOC(*model_buffer, *buffer_capacity, sizeof(struct gsgp_tree *));
    }

    (*model_buffer)[N++] = model;

    return N;
}

static int extract_models(struct gsgp_individual *ind,
                               struct gsgp_tree ***model_buffer, int N, int *buffer_capacity)
{
    if (ind->type == MUTANT) {
        N = extract_models(ind->details.mutation.base, model_buffer, N, buffer_capacity);
        N = extract_models(ind->details.mutation.mut, model_buffer, N, buffer_capacity);
    } else if (ind->type == OFFSPRING) {
        N = extract_models(ind->details.parents.mother, model_buffer, N, buffer_capacity);
        N = extract_models(ind->details.parents.father, model_buffer, N, buffer_capacity);
    } else { /* the individual must be a base model */
        N = add_model(ind->details.base.model, model_buffer, N, buffer_capacity);
    }

    return N;
}

static double *extract_coef(struct gsgp_individual *ind, struct gsgp_tree **models, int N)
{
    double *coef_a, *coef_b;
    int i;

    if (ind->type == BASE) {
        coef_a = malloc(N * sizeof(double));
        for (i = 0; i < N; ++i) coef_a[i] = (models[i] == ind->details.base.model) ? 1.0 : 0.0;
    } else if (ind->type == OFFSPRING) {
        coef_a = extract_coef(ind->details.parents.mother, models, N);
        coef_b = extract_coef(ind->details.parents.father, models, N);

        for (i = 0; i < N; ++i) coef_a[i] *= ind->details.parents.pm;
        for (i = 0; i < N; ++i) coef_b[i] *= ind->details.parents.pf;
        for (i = 0; i < N; ++i) coef_a[i] += coef_b[i];
        free(coef_b);
    } else {
        coef_a = extract_coef(ind->details.mutation.base, models, N);
        coef_b = extract_coef(ind->details.mutation.mut, models, N);

        for (i = 0; i < N; ++i) coef_a[i] += ind->details.mutation.ms * coef_b[i];
        free(coef_b);
    }

    return coef_a;
}
/***********************************************************************************************************************
 * END HELPER ROUTINES
 **********************************************************************************************************************/
