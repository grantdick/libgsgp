#ifndef _GSGP_ALLOC_H
#define	_GSGP_ALLOC_H

#ifdef	__cplusplus
extern "C" {
#endif

    #include <stdbool.h>
    #include <string.h>

    #define ALLOC(n, sz, clear) reallocate(NULL, (n), (sz), __FILE__, __LINE__, clear)
    #define REALLOC(base, n, sz) reallocate((base), (n), (sz), __FILE__, __LINE__, false)
    static void *reallocate(void *base, int n, size_t sz, char *src_file, int line, bool clear)
    {
        base = realloc(base, n * sz);
        if (base == NULL) {
            fprintf(stderr, "%s:%d - ERROR: Failed to allocate memory\n",
                    src_file, line);
            exit(EXIT_FAILURE);
        }

        if (clear) memset(base, 0, n * sz);

        return base;
    }

#ifdef	__cplusplus
}
#endif

#endif
