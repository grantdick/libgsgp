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
        /* this defines the behaviour of the search mechanism:
         * HILLCLIMB defines a single-point steepest descent search
         * algorithm, while POPULATION_SEARCH defines a search model
         * similar to Koza's original GP specification */
        enum {
            HILLCLIMB,
            POPULATION_SEARCH
        } search_model;

        /* this defines the underlying representation used in the
         * model. Several options are available:
         *
         * STD_PARSE_TREE      - essentially, the standard parse tree
         *                       representation used by Koza and
         *                       others, here the functions are +, -,
         *                       *, and a (safe) /, while the
         *                       terminals are the independent
         *                       variables of the data set
         *
         * NODIV_PARSE_TREE    - as above, but without the protected
         *                       division operator (we've found that
         *                       almost all "catastrophic" overfitting
         *                       events are due to the presence of the
         *                       division operator, and division by
         *                       "near zero", which is not easily
         *                       managed by the protected operator)
         *
         * INTERVAL_PARSE_TREE - as for STD_PARSE_TREE, but "safe"
         *                       tree initialisation is used, where
         *                       the known execution intervals of
         *                       subtrees are used to pick the
         *                       function at a given node
         *
         * REGRESSION_TREE     - replaces the parse tree representation
         *                       completely, and uses a regression
         *                       tree-like model (i.e., split nodes,
         *                       and region terminals). The tree
         *                       construction algorithm is vary simple
         *                       and naive, and does not attempt any
         *                       information gain-like operators to
         *                       select split nodes
         */
        enum {
            STD_PARSE_TREE,
            NODIV_PARSE_TREE,
            INTERVAL_PARSE_TREE,
            REGRESSION_TREE
        } tree_model;

        /* determines the coefficients for each parent when performing
         * crossover. Several options:
         *
         * FIXED_PROPORTION - equal contribution of parents (i.e.,
         *                    coef = 0.5)
         *
         * RANDOM_PROPORTION - SGXE as defined by Moraglio (i.e, p and
         *                     (1 - p) determined from uniform sample
         *                     of [0, 1))
         *
         * LEAST_SQUARES_CROSSOVER - coefficients of parents are p and
         *                           (1 - p), with p determined by
         *                           least squares estimation to
         *                           minimise the residuals of the
         *                           model
         *
         * MULTIPLE_LEAST_SQUARES_CROSSOVER - as above, but two
         *                                    degrees of freedom
         *                                    instead of one (i.e.,
         *                                    two coefficients
         *                                    determined through
         *                                    ordinary least squares
         *                                    estimation)
         */
        enum {
            FIXED_PROPORTION,
            RANDOM_PROPORTION,
            LEAST_SQUARES_CROSSOVER,
            MULTIPLE_LEAST_SQUARES_CROSSOVER
        } crossover_coef;

        /* determines the behaviour of mutation coefficients. Several
         * options are available:
         *
         * FIXED_SCALE - as defined by Moraglio, ms is a fixed value
         *               (e.g., 0.1) used for all mutations
         *
         * RANDOM_SCALE - as proposed by Vanneschi et al, ms is
         *                determined from uniform sampling of [0, 1)
         *
         * LEAST_SQUARES_MUTATION - as for crossover, ms is determined
         *                          through least squares estimation
         *                          to minimise the residuals of the
         *                          model
         */
        enum {
            FIXED_SCALE,
            RANDOM_SCALE,
            LEAST_SQUARES_MUTATION
        } mutation_coef;

        /* balanced mutation (i.e., the mutant is rooted with a
         * subtraction node, so that the resultant mutation is R1 -
         * R2, where R1 and R2 are random trees). If set to false,
         * then the mutant tree is generated as per a standard tree
         * initialisation */
        bool balanced_mutation;

        /* logistic mutation (proposed by Castelli et al) wraps the
         * each mutant tree (or subtree when balanced mutation is
         * used) with a logistic function to bound its output to [0,1]
         * (or [-1,1] in the case of balanced mutation) */
        bool logistic_mutation;

        double ms;      /* mutation scale used with FIXED_SCALE */

        int N; /* population size */
        int G; /* number of generations */

        double p_cross; /* probability of crossover */
        double p_mut;   /* probability of mutation */

        int tourn_k;    /* size of tournament used in population
                         * search method */

        int min_depth;  /* minimum tree depth */
        int max_depth;  /* maximum tree depth */

        int n_features; /* number of independent variables - this is
                         * usually set programmatically */

        double **feature_interval; /* the known intervals of the
                                    * independent variables in the
                                    * problem - are usually set
                                    * through examination of training
                                    * data, but can incorporate any
                                    * domain knowledge as
                                    * required. The resulting
                                    * intervals are only used when
                                    * safe tree initialisation is
                                    * required */

        double (*rnd)(void); /* the random number generator of the
                              * system, returns a floating point
                              * number in [0, 1). If none supplied,
                              * then the stdlib function rand()
                              * function is used (divided by (RAND_MAX
                              * + 1.0)) */
    };

    struct gsgp_tree; /* opaque structure for parse/regression tree details */

    struct gsgp_individual {
        /* labels the type of operation used to create the individual
         * (BASE = initialisation, OFFSPRING = crossover, MUTANT =
         * mutation) */
        enum { BASE, OFFSPRING, MUTANT } type;

        /* the various bits of book-keeping required for each
         * individual resulting from different operations */
        union {
            /* initialisation */
            struct {
                int model_id;
                struct gsgp_tree *model;
            } base;

            /* crossover */
            struct {
                struct gsgp_individual *mother;
                struct gsgp_individual *father;
                double pm; /* coefficient of mother */
                double pf; /* coefficient of father */
            } parents;

            /* mutation */
            struct {
                struct gsgp_individual *base;
                struct gsgp_individual *mut;
                double ms; /* mutation scale */
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

        double size;        /* the total number of nodes
                             * (incl. references to parent trees) used
                             * by the individual */

        bool referenced;    /* true if individual is in the current
                             * population, or is an ancestor of an
                             * individual within the current
                             * population - used internally by the
                             * archive pruning process, and can be
                             * ignored outside of the framework */

        double fitness;     /* the fitness assigned to the individual
                             * after evaluation */
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

    /* returns a parameters structure with the following values:
     *
     *   search_model = POPULATION_SEARCH;
     *   tree_model = REGRESSION_TREE;
     *   crossover_coef = LEAST_SQUARES_CROSSOVER;
     *   mutation_coef = FIXED_SCALE;
     *   balanced_mutation = false;
     *   logistic_mutation = false;
     *   N = 200;
     *   G = 1000;
     *   p_cross = 0.3;
     *   p_mut = 0.6;
     *   tourn_k = 4;
     *   ms = 0.1;
     *   min_depth = 2;
     *   max_depth = 6;
     *   n_features = -1;
     *   feature_interval = NULL;
     *   rnd = default_rnd; (i.e., (double)rand() / (double)(RAND_MAX + 1.0))
     *
     * the values for many of these parameters are largely taken from
     * previous work (e.g., Castelli et al, GP & EM 15(1) 2014), with
     * some exceptions (e.g., the regression tree representation is
     * used by default)
     *
     * RETURN VALUE:
     * a pointer to a parameter structure with the specified default
     * values
     */
    struct gsgp_parameters *gsgp_default_parameters();

    /* frees the memory used by the parameters structure pointed to by
     * the supplied parameter
     *
     * PARAMETERS:
     * param - pointer to the system parameters structure
     */
    void gsgp_release_parameters(struct gsgp_parameters *param);

    /* updates the supplied parameters to indicate the number of
     * independent variables (features) in the data set, and the known
     * intervals for these features (which can be NULL if safe
     * initialisation is not used)
     *
     * PARAMETERS:
     * param - pointer to the system parameters structure
     *
     * n_features - the number of independent features in the data
     *
     * intervals - a array (size [n_features][2]) that contains the
     *             upper and lower bounds of each feature
     */
    void gsgp_set_features(struct gsgp_parameters *param, int n_features, double **intervals);

    /* sets the system-wide random number generator
     * PARAMETERS:
     * param - pointer to the system parameters structure
     *
     * rnd - function pointer that takes no arguments and returns a
     *       floating point value in [0,1)
     */
    void gsgp_set_rng(struct gsgp_parameters *param, double (*rnd)(void));

    /* the archive is used for memory management. As the system uses
     * pointers to parents, rather than deep copying, there is a need
     * to keep and update an archive of previously generated
     * individuals. Essentially, this behaves like reference counting
     * in some garbage collection systems, although the memory
     * management is handled manually in the system
     *
     * PARAMETERS:
     * initial_capacity - the initial capacity of the archive (i.e.,
     *                    how many individuals we expect to keep),
     *                    this is an initial limit that will grow as
     *                    needed. Typically, this is set to N*G
     *                    individuals (where N is population size, and
     *                    G is number of generations).
     *
     * n_instances - the number of data instances in the data
     *               set. When an individual is entered into the
     *               archive, its execution cache (yhat) is
     *               initialised to this size, and the result of its
     *               execution is stored to speed up future
     *               evaluations
     *
     * RETURN VALUE:
     * a pointer to an archive structure
     */
    struct gsgp_archive *gsgp_create_archive(int initial_capacity, int n_instances);

    /* prunes the archive to remove any individuals not referenced by
     * the individuals present in current. This usually done after a
     * new generation is created
     *
     * PARAMETERS:
     * archive - the system archive to be pruned
     *
     * current - a pointer to the current population, any individual
     *           in the archive not directly in current, or not
     *           referenced by members of current, will be removed
     *
     * N - the size of current
     */
    void gsgp_prune_archive(struct gsgp_archive *archive, struct gsgp_individual **current, int N);

    /* cleans up any resources used by the archive (including any individuals in the archive)
     *
     * PARAMETERS:
     * archive - the system archive to clean up
     */
    void gsgp_release_archive(struct gsgp_archive *archive);


    /* runs the system to completion under the specified parameters
     *
     * PARAMETERS:
     * param   - the system parameters
     * archive - the system archive
     * X       - the independent variables of the training data
     * Y       - the response variable of the training data
     *
     * idx     - an array of training instance ids (used for
     *           performance caching). Can be set to NULL if no
     *           caching is to be used, but the performance hit is
     *           substantial
     * n       - the number of training instances

     * EVAL    - pointer to callback function to evaluate an
     *           individual, resulting call is then:
     *              EVAL(individual, X, Y, idx, n, param, opt)
     *
     * PRINT   - pointer to callback function to print end-of iteration
     *           output, resulting call is then:
     *              PRINT(pop, N, generation, best_idx, param, archive, opt)
     *
     * opt     - pointer to a structure that contains additional arguments
     *           used in the EVAL and/or PRINT callbacks
     *
     * RETURN VALUE:
     * a pointer to the best individual in the final iteration
     */
    struct gsgp_individual *gsgp_run(struct gsgp_parameters *param, struct gsgp_archive *archive,
                                     double **X, double *Y, int *idx, int n,
                                     double (*EVAL)(struct gsgp_individual *, double **, double *, int *, int,
                                                    struct gsgp_parameters *, void *),
                                     void   (*PRINT)(struct gsgp_individual **, int, int, int,
                                                     struct gsgp_parameters *, struct gsgp_archive *, void *),
                                     void   *opt);

    /* creates a BASE individual using the supplied parameters. Most
     * applications will not call this function directly, but will
     * rather use its behaviour within the gsgp_run() function. Only
     * applications that define "custom" behaviour will need to call
     * this function directly
     *
     * PARAMETERS:
     *
     * min_depth - the minimum required tree depth
     *
     * max_depth - the maximum allowed tree depth
     *
     * X         - the independent variables of the training data (only
     *             required for building regression trees, can be NULL
     *             otherwise)
     *
     * Y         - the response variable of the training data
     *
     * n         - the number of instances in the training data
     *
     * param     - pointer to the global system parameters
     *
     * archive   - the current system archive
     *
     * RETURN VALUE:
     * a pointer to a new BASE individual
     */
    struct gsgp_individual *gsgp_create_individual(int min_depth, int max_depth,
                                                   double **X, double *Y, int n,
                                                   struct gsgp_parameters *param,
                                                   struct gsgp_archive *archive);

    /* creates an OFFSPRING individual using the supplied
     * parameters. Most applications will not call this function
     * directly, but will rather use its behaviour within the
     * gsgp_run() function. Only applications that define "custom"
     * behaviour will need to call this function directly
     *
     * PARAMETERS:
     *
     * mother - pointer to the first parent of crossover
     *
     * father - pointer to the second parent of crossover
     *
     * X      - the independent variables of the training data (only
     *          required for building regression trees, can be NULL
     *          otherwise)
     *
     * ids    - the ids for the supplied training instances (for
     *          lookup and caching purposes)
     *
     * Y      - the response variable of the training data
     *
     * n      - the number of instances in the training data
     *
     * param  - pointer to the global system parameters
     *
     * archive - the current system archive
     *
     * RETURN VALUE:
     * a pointer to a new OFFSPRING individual
     */
    struct gsgp_individual *gsgp_crossover(struct gsgp_individual *mother, struct gsgp_individual *father,
                                           double **X, double *Y, int *ids, int n,
                                           struct gsgp_parameters *param, struct gsgp_archive *archive);

    /* creates a MUTANT individual using the supplied parameters. Most
     * applications will not call this function directly, but will
     * rather use its behaviour within the gsgp_run() function. Only
     * applications that define "custom" behaviour will need to call
     * this function directly
     *
     * PARAMETERS:
     *
     * base   - pointer to the individual from which mutation is based
     * X      - the independent variables of the training data (only
     *          required for building regression trees, can be NULL
     *          otherwise)
     * ids    - the ids for the supplied training instances (for
     *          lookup and caching purposes)
     * Y      - the response variable of the training data
     * n      - the number of instances in the training data
     * param  - pointer to the global system parameters
     * archive - the current system archive
     *
     * RETURN VALUE:
     * a pointer to a new MUTANT individual
     */
    struct gsgp_individual *gsgp_mutation(struct gsgp_individual *base,
                                          double **X, double *Y, int *ids, int n,
                                          struct gsgp_parameters *param, struct gsgp_archive *archive);

    /* executes the individual on the supplied features and returns
     * the value of execution. This will most likely be called within
     * the EVAL callback supplied in gsgp_run(), or used within the
     * external application once the "best" model has been established
     *
     * PARAMETERS:
     *
     * ind      - pointer to the individual to execute
     *
     * id       - the unique identifier assigned to this instance.
     *            If the individual has been previously executed, then
     *            this value is used to look up the previous execution
     *            result. Otherwise, the instance is executed, and the
     *            result is stored under this id. If the supplied id
     *            is less than zero, then no caching takes place
     *
     * features - the independent variable values for the current
     *            instance
     *
     * param    - pointer to the global system parameters
     *
     * RETURN VALUE:
     * the resulting value of execution
     */
    double gsgp_exec_individual(struct gsgp_individual *ind, int id, double *features, struct gsgp_parameters *param);

    /* executes the supplied tree model on the supplied data and
     * returns the result. Most applications should not have a need to
     * call this function, and instead should call
     * gsgp_exec_individual()
     *
     * PARAMETERS:
     *
     * model - pointer to the tree to execute
     *
     * data  - the independent variable values for the current
     *         instance
     *
     * RETURN VALUE:
     * the resulting value of execution
     */
    double gsgp_execute_model(struct gsgp_tree *model, double *data);

    /* prints the specified individual to the specified output
     * stream. Individuals are printed out in an inline language that
     * should be easily executed in C or R without significant
     * trouble.
     *
     * PARAMETERS:
     *
     * ind      - pointer to the individual to print
     *
     * out      - the file stream to which output is directed
     */
    void   gsgp_print_individual(struct gsgp_individual *ind, FILE *out);

    /* prints the supplied tree model to the supplied file
     * stream. Most applications should not have a need to call this
     * function, and instead should call gsgp_print_individual()
     *
     * PARAMETERS:
     *
     * model - pointer to the tree to print
     *
     * out   - the file stream to which output is directed
     *
     * RETURN VALUE:
     * the resulting value of execution
     */
    void   gsgp_print_model(struct gsgp_tree *model, FILE *out);





    /******************************************************************
     * THE FOLLOWING FUNCTIONS ARE ... EXPERIMENTAL. ESSENTIALLY, THEY
     * ARE TRYING TO CAPTURE THE UNDERLYING RELATIONSHIP BETWEEN GSGP
     * AND ADDITIVE MODELS (ESP. BOOSTING METHODS) BY DEMONSTRATING
     * THAT THE MODELS BUILT BY GSGP CAN BE EASILY CONVERTED INTO
     * ENSEMBLES AND EXECUTED FROM THERE. HOWEVER, THE EMPHASIS OF
     * THIS RELATIONSHIP IS NOW THE FOCUS OF A SEPARATE PROJECT SO,
     * ALTHOUGH THESE WORK, THEY WILL NOT RECEIVE ANY FURTHER
     * ATTENTION
     *
     * the first function, takes an individual and the associated
     * archive, and determines the size of the equivalent ensemble
     * when viewed as an additive model
     *
     * the second function converts the "tree" of the supplied model
     * into two components, the ensemble of models embedded in the
     * tree, and the corresponding coefficients of the models
     ******************************************************************/
    int    gsgp_ensemble_size(struct gsgp_individual *ind, struct gsgp_archive *archive);

    int    gsgp_extract_model(struct gsgp_individual *ind, double **beta_ptr, struct gsgp_tree ***models_ptr,
                              struct gsgp_archive *archive);

#ifdef __cplusplus
}
#endif

#endif
