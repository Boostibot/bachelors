#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#ifndef _cplusplus
#include <stdbool.h>
#endif

typedef double Real;

bool is_near(Real a, Real b, Real epsilon)
{
    //this form guarantees that is_nearf(NAN, NAN, 1) == true
    return !(fabs(a - b) > epsilon);
}

//Returns true if x and y are within epsilon distance of each other.
//If |x| and |y| are less than 1 uses epsilon directly
//else scales epsilon to account for growing floating point inaccuracy
bool is_near_scaled(Real x, Real y, Real epsilon)
{
    //This is the form that produces the best assembly
    Real calced_factor = fabs(x) + fabs(y);
    Real factor = 2 > calced_factor ? 2 : calced_factor;
    return is_near(x, y, factor * epsilon / 2);
}

bool are_near_scaled(const Real* a, const Real* b, int n, Real epsilon)
{
    for(int i = 0; i < n; i++)
        if(is_near_scaled(a[i], b[i], epsilon) == false)
            return false;

    return true;
}

#include <stdarg.h>
void matrix_print(const Real* matrix, int height, int width, const char* format, ...)
{
    if(format != NULL && strlen(format) > 0)
    {
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
        printf("\n");
    }

    for(int y = 0; y < height; y++)
    {
        printf("[");
        for(int x = 0; x < width; x++)
        {
            Real item = matrix[x + y*width];
            if(item < 0)
            {
                item = -item;
                printf("-");
            }
            else
                printf(" ");
                
            printf("%.6f ", item);
        }
        printf("]\n");
    }
}

void matrix_invert_in_out(Real* out, Real* in, int n)
{
    const Real EPSILON = (Real) 0.0000001;
    assert(out != NULL && in != NULL && n >= 0);

    memset(out, 0, sizeof(Real) * (size_t)n* (size_t)n);
    for(int i = 0; i < n; i++)
        out[i + i*n] = (Real) 1;

    for(int y = 0; y < n; y++)
    {
        Real pivot = in[y + y*n];
        in[y + y*n] = (Real) 1;
        out[y + y*n] = (Real) 1/pivot;

        assert(is_near(pivot, 0, EPSILON) == 0 && "we dont do swapping of rows because we are lazy");
        for(int x = y + 1; x < n; x++)
            in[x + y*n] /= pivot;

        //@NOTE: future me if you are going to use this code to solve
        // Ax = B where B is general vector or series of vectors dont
        // forget to change the iteration bound here since out is no 
        // longer diagonal! Also do this for the other loop!
        for(int x = 0; x < y; x++)
            out[x + y*n] /= pivot;

        //Iterate through all rows below y
        // (read y_ as y')
        for(int y_ = y + 1; y_ < n; y_++ )
        {
            Real factor = in[y + y_*n];

            //Substract from the y' row below a multiple of the y row
            // to zero the first element 
            // (we explicitly set the first element to zero to be more stable)
            in[y + y_*n] = (Real) 0;
            out[y + y_*n] -= out[y + y*n] * factor;

            if(is_near(factor, 0, EPSILON))
                continue;

            for(int x = y + 1; x < n; x++ )
                in[x + y_*n] -= in[x + y*n] * factor;

            for(int x = 0; x < y; x++)
                out[x + y_*n] -= out[x + y*n] * factor; 
        }

    }

    #ifndef NDEBUG
    for(int y = 0; y < n; y++)
    {
        for(int x = 0; x < y; x++)
            assert(is_near_scaled(in[x + y*n], 0, EPSILON) && "Should be trinagular!");

        assert(is_near_scaled(in[y + y*n], 1, EPSILON) && "Must have 1 on diagonal");
    }
    #endif // !NDEBUG

    for(int y = n; y-- > 1; )
    {
        for(int y_ = y; y_-- > 0; )
        {
            Real factor = in[y + y_*n];
            in[y + y_*n] = 0;

            if(is_near(factor, 0, EPSILON))
                continue;

            for(int x = 0; x < n; x++ )
            {
                out[x + y_*n] -= out[x + y*n] * factor;    
            }
        }
    }
    
    #ifndef NDEBUG
    for(int y = 0; y < n; y++)
    {
        for(int x = 0; x < n; x++)
        {
            Real should_be = x == y ? 1 : 0;
            assert(is_near_scaled(in[x + y*n], should_be, EPSILON) && "Should be identity!");
        }
    }
    #endif // !NDEBUG
}

#include <stdlib.h>
void matrix_invert(Real* out, const Real* in, int n)
{
    //@NOTE: lazy, bad programmer static data cache to make our life easier.
    // (its simpler to call and use)
    static int static_n = 0;
    static Real* static_data = NULL;
    size_t byte_size = (size_t) n * (size_t) n * sizeof(Real);
    if(static_n < n)
    {
        Real* new_static = (Real*) realloc(static_data, byte_size);
        assert(new_static != NULL && "if realloc fails there is very little we can do");
        static_data = new_static;
        static_n = n;
    }

    memcpy(static_data, in, byte_size);
    matrix_invert_in_out(out, static_data, n);
}

void matrix_multiply(Real* output, const Real* A, const Real* B, int A_height, int A_width, int B_height, int B_width)
{
    assert(A_width == B_height);
    for(int y = 0; y < A_height; y++)
    {
        for(int x = 0; x < B_width; x++)
        {
            Real val = 0;
            for(int k = 0; k < A_width; k++)
            {
                Real a = A[k + y*A_width];
                Real b = B[x + k*B_width];
                val += a*b;
            }

            output[x + y*B_width] = val;
        }
    }
}

Real vector_dot_product(const Real* a, const Real* b, int n)
{
    Real out = 0;
    matrix_multiply(&out, a, b, 1, n, n, 1);
    return out;
}

typedef struct Conjugate_Gardient_Params {
    Real epsilon;
    Real tolerance;
    int max_iters;
    bool padded;
} Conjugate_Gardient_Params;

bool matrix_conjugate_gradient_solve(const Real* A, Real* x, const Real* b, int N, const Conjugate_Gardient_Params* params_or_null)
{
    Conjugate_Gardient_Params params = {0};
    params.epsilon = (Real) 1.0e-10;
    params.tolerance = (Real) 1.0e-5;
    params.max_iters = 10;
    params.padded = false;

    if(params_or_null)
        params = *params_or_null;
    
    size_t vec_byte_size = sizeof(Real)*(size_t)N;


    //@NOTE: Evil programmer doing evil programming practices
    static int static_cap = 0;
    static Real* r = NULL;
    static Real* p = NULL;
    static Real* Ap = NULL;
    if(static_cap < N)
    {
        r = (Real*) realloc(r, vec_byte_size);
        p = (Real*) realloc(p, vec_byte_size);
        Ap = (Real*) realloc(Ap, vec_byte_size);
    }

    memset(x, 0, vec_byte_size);
    memcpy(r, b, vec_byte_size);
    memcpy(p, b, vec_byte_size);
    
    #define MAX(a, b)   ((a) > (b) ? (a) : (b))

    Real r_dot_r = vector_dot_product(r, r, N);
    int iter = 0;
    for(; iter < params.max_iters; iter++)
    {
        matrix_multiply(Ap, A, p, N, N, N, 1);
        
        Real p_dot_Ap = vector_dot_product(p, Ap, N);
        Real alpha = r_dot_r / MAX(p_dot_Ap, params.epsilon);
        
        for(int i = 0; i < N; i++)
        {
            x[i] = x[i] + alpha*p[i];
            r[i] = r[i] - alpha*Ap[i];
        }

        Real r_dot_r_new = vector_dot_product(r, r, N);
        if(r_dot_r_new/N < params.tolerance*params.tolerance)
            break;

        Real beta = r_dot_r_new / MAX(r_dot_r, params.epsilon);
        for(int i = 0; i < N; i++)
        {
            p[i] = r[i] + beta*p[i]; 
        }

        r_dot_r = r_dot_r_new;
    }
    #undef MAX
    printf("Solved with %i iters\n", iter);
    return iter != params.max_iters;
}

typedef struct Cross_Matrix {
    Real C;
    Real U;
    Real D;
    Real L;
    Real R;
} Cross_Matrix;

enum {
    OFF_DIAGONAL_EXACT = 0,
    OFF_DIAGONAL_PADDED = 1,
};

void cross_matrix_multiply(Real* out, const Cross_Matrix A, const Real* x, int n, int m, bool is_padded)
{
    int N = n*m;
    if(is_padded)
    {
        for(int i = 0; i < N; i ++)
        {
            Real val = 0;
            val += x[i]*A.C;
            //@NOTE: No edge logic! We require explicit (m) padding to be added on both sides of x!
            val += x[i+1]*A.R;
            val += x[i-1]*A.L;
            val += x[i+m]*A.U;
            val += x[i-m]*A.D;

            out[i] = val;
        }
    }
    else
    {
        for(int i = 0; i < N; i ++)
        {
            Real val = 0;
            val += x[i]*A.C;
            if(i+1 < N) 
                val += x[i+1]*A.R;
            if(i-1 >= 0) 
                val += x[i-1]*A.L;
            if(i+m < N) 
                val += x[i+m]*A.U;
            if(i-m >= 0) 
                val += x[i-m]*A.D;

            out[i] = val;
        }
    }
}

bool cross_matrix_conjugate_gradient_solve(Cross_Matrix A, Real* x, const Real* b, int n, int m, const Conjugate_Gardient_Params* params_or_null)
{
    Conjugate_Gardient_Params params = {0};
    params.epsilon = (Real) 1.0e-10;
    params.tolerance = (Real) 1.0e-5;
    params.max_iters = 10;
    params.padded = false;
    if(params_or_null)
        params = *params_or_null;
    
    int N = n*m;
    int halo = m;
    size_t vec_byte_size = sizeof(Real)*(size_t)N;

    //@NOTE: Evil programmer doing evil programming practices
    static int static_cap = 0;
    static Real* r = NULL;
    static Real* _p = NULL;
    static Real* Ap = NULL;
    if(static_cap < N)
    {
        r =  (Real*) realloc(r, vec_byte_size);
        _p = (Real*) realloc(_p, sizeof(Real)*((size_t)N + 2*(size_t)halo)); //@NOTE: No edge logic! We require explicit (m) padding to be added on both sides of p!
        Ap = (Real*) realloc(Ap, vec_byte_size);
    }

    Real* p = _p + halo;
    if(params.padded)
    {
        memset(p - halo, 0, sizeof(Real)*(size_t)halo);
        memset(p + N, 0, sizeof(Real)*(size_t)halo);
    }

    memset(x, 0, vec_byte_size);
    memcpy(r, b, vec_byte_size);
    memcpy(p, b, vec_byte_size);
    
    #define MAX(a, b)   ((a) > (b) ? (a) : (b))

    Real r_dot_r = vector_dot_product(r, r, N);
    int iter = 0;
    for(; iter < params.max_iters; iter++)
    {
        cross_matrix_multiply(Ap, A, p, n, m, params.padded);

        Real p_dot_Ap = vector_dot_product(p, Ap, N);
        Real alpha = r_dot_r / MAX(p_dot_Ap, params.epsilon);
        
        for(int i = 0; i < N; i++)
        {
            x[i] = x[i] + alpha*p[i];
            r[i] = r[i] - alpha*Ap[i];
        }

        Real r_dot_r_new = vector_dot_product(r, r, N);
        if(r_dot_r_new/N < params.tolerance*params.tolerance)
            break;

        Real beta = r_dot_r_new / MAX(r_dot_r, params.epsilon);
        for(int i = 0; i < N; i++)
            p[i] = r[i] + beta*p[i]; 

        r_dot_r = r_dot_r_new;
    }
    #undef MAX

    printf("Solved with %i iters\n", iter);
    return iter != params.max_iters;
}

bool matrix_is_identity(const Real* matrix, Real epsilon, int n)
{
    for(int y = 0; y < n; y++)
        for(int x = 0; x < n; x++)
        {
            Real should_be = x == y ? 1 : 0;
            Real is_actually = matrix[x + y*n];
            if(is_near_scaled(should_be, is_actually, epsilon) == false)
                return false;
        }

    return true;
}

//Enforces x to be true. Other arguments may be supplied and will be printed.
#define TEST(x, ...) ((x) ? (void) 0 : (printf("test %s failed!\n", #x ), printf("-> " __VA_ARGS__), abort()))

int main()
{

    Conjugate_Gardient_Params params = {0};
    params.epsilon = (Real) 1.0e-10;
    params.tolerance = (Real) 1.0e-5;
    params.max_iters = 10;
    params.padded = false;
    {
        enum {N = 3, M = 4};
        Real A[N*N] = {
            3, 1, 0,
            1, 3, 1,
            0, 1, 3,
        };
        Real B[N*M] = {
            0, 2, 1, 3,
            0, 3, 0, 0,
            8, 5, 1, 1,
        };

        Real expected[N*M] = {
            0,	9,	3,	9,
            8,	16,	2,	4,
            24,	18,	3,	3,
        };
        Real AB[N*M] = {0};
        matrix_multiply(AB, A, B, N, N, N, M);
        TEST(are_near_scaled(AB, expected, N*M, 0.00000001f));
    }

    {
        enum {N = 3};
        Real matrix[N*N] = {
            3, 1, 0,
            1, 3, 1,
            0, 1, 3,
        };

        Real inverse[N*N] = {0};
        Real identity[N*N] = {0};
        matrix_invert(inverse, matrix, N);
        matrix_multiply(identity, matrix, inverse, N, N, N, N);
        assert(matrix_is_identity(identity, 0.00001f, N));
        
        //test multiply more than the actual inverse
        matrix_multiply(matrix, identity, inverse, N, N, N, N);
        assert(matrix_is_identity(identity, 0.00001f, N));
    }
    {
        enum {N = 4};
        Real matrix[N*N] = {
            4, 1, 0, 0,
            1, 4, 1, 0,
            0, 1, 4, 1,
            0, 0, 1, 4,
        };

        Real inverse[N*N] = {0};
        Real identity[N*N] = {0};
        matrix_invert(inverse, matrix, N);
        matrix_multiply(identity, matrix, inverse, N, N, N, N);
        assert(matrix_is_identity(identity, 0.00001f, N));
    }
    {
        enum {N = 2};
        Real A[N*N] = {
            4, 1,
            1, 3,
        };

        Real b[N] = {1, 2};
        Real x[N] = {0};

        TEST(matrix_conjugate_gradient_solve(A, x, b, N, &params), "must converge within max iters!");

        Real should_be_b[N] = {0};
        matrix_multiply(should_be_b, A, x, N, N, N, 1);
        TEST(are_near_scaled(b, should_be_b, N, params.tolerance));
    } 
    {
        enum {N = 4};
        Real A[N*N] = {
            4, 1, 0, 0,
            1, 4, 1, 0,
            0, 1, 4, 1,
            0, 0, 1, 4,
        };

        Real b[N] = {1, 2, 3, 4};
        Real x[N] = {0};

        TEST(matrix_conjugate_gradient_solve(A, x, b, N, &params), "must converge within max iters!");

        Real should_be_b[N] = {0};
        matrix_multiply(should_be_b, A, x, N, N, N, 1);
        TEST(are_near_scaled(b, should_be_b, N, params.tolerance));
    }
    {
        enum {m = 2, n = 2, N = m*n};
        //This is the exact same case as with A above
        Cross_Matrix A = {0};
        A.C = 4;
        A.L = 1;
        A.R = 1;

        Real b[N] = {1, 2, 3, 4};
        Real x[N] = {0};

        TEST(cross_matrix_conjugate_gradient_solve(A, x, b, n, m, &params), "must converge within max iters!");

        Real should_be_b[N] = {0};
        cross_matrix_multiply(should_be_b, A, x, n, m, false);
        TEST(are_near_scaled(b, should_be_b, N, params.tolerance));
    }

    {
        Conjugate_Gardient_Params padded_params = params;
        padded_params.padded = true;

        enum {m = 2, n = 2, N = m*n, H = m};
        Cross_Matrix A = {0};
        A.C = 4;
        A.L = 1;
        A.R = 1;
        A.U = 1;
        A.D = 1;

        Real _b[N+2*H] = {0};
        Real _x[N+2*H] = {0};
        Real _Ax[N+2*H] = {0};

        //Fill with dummy data to see if the function memsets the halo correctly!
        memset(_b, 0x55, sizeof(_b));

        Real* b = _b + H;
        Real* x = _x + H;
        Real* Ax = _Ax + H;

        for(int y = 0; y < n; y++)
            for(int x = 0; x < m; x++)
                b[x + y*m] = x + y;

        TEST(cross_matrix_conjugate_gradient_solve(A, x, b, n, m, &padded_params), "must converge within max iters!");

        cross_matrix_multiply(Ax, A, x, n, m, padded_params.padded);
        TEST(are_near_scaled(b, Ax, N, params.tolerance));
    }
    
    {
        enum {
            m = 32,
            n = 32,
            N = m*n, 
            H = m,
            MATRIX = N*N*sizeof(Real),
            VECTOR = N*sizeof(Real),
            PADDED_VECTOR = (N + 2*H)*sizeof(Real),
        };
        const Real diag = 4;
        const Real off = 1;

        {
            Cross_Matrix A = {diag, off, off, off, off};
            
            Real* _b =    (Real*) malloc(PADDED_VECTOR);
            Real* _x =    (Real*) malloc(PADDED_VECTOR);
            Real* _Ax =   (Real*) malloc(PADDED_VECTOR);
            TEST(_b && _x && _Ax, "out of memory!");

            //Fill with dummy data to see if the function memsets the halo correctly!
            memset(_b, 0x55, PADDED_VECTOR);
            //But zero the rest. 
            memset(_x, 0, PADDED_VECTOR);
            memset(_Ax, 0, PADDED_VECTOR);

            Real* b = _b + H;
            Real* x = _x + H;
            Real* Ax = _Ax + H;

            for(int y = 0; y < n; y++)
                for(int x = 0; x < m; x++)
                    b[x + y*m] = x + y;

            Conjugate_Gardient_Params padded_params = params;
            padded_params.max_iters = 100;
            padded_params.padded = true;

            printf("solving %ix%i cross matrix...\n", N, N);
            TEST(cross_matrix_conjugate_gradient_solve(A, x, b, m, m, &padded_params), "must converge within max iters!");
            printf("done!\n");

            cross_matrix_multiply(Ax, A, x, n, m, true);
            TEST(are_near_scaled(Ax, b, N, padded_params.tolerance*10));

            free(_b);
            free(_x);
            free(_Ax);
        }
        
        {
            Real* A =    (Real*) malloc(MATRIX);
            Real* Ainv = (Real*) malloc(MATRIX);
            Real* I =    (Real*) malloc(MATRIX);
            Real* b =    (Real*) malloc(VECTOR);
            Real* x =    (Real*) malloc(VECTOR);
            Real* Ax =   (Real*) malloc(VECTOR);
        
            TEST(A && Ainv && I && b && x, "out of memory!");
            //This type of A appears in finite-volume/finite-difference schemes
            // for regular grid when approximating laplacians. As thats what I am ultimately doing
            // its good to test if it actually works.
            for(int y = 0; y < N; y++)
            {
                for(int x = 0; x < N; x++)
                {
                    if(x == y)          A[x + y*N] = diag;
                    else if(x == y + 1) A[x + y*N] = off;
                    else if(x == y - 1) A[x + y*N] = off;
                    else if(x == y + m) A[x + y*N] = off;
                    else if(x == y - m) A[x + y*N] = off;
                    else                A[x + y*N] = 0;
                }
            }

            //b is a vector of size N but can be viewed as matrix of
            // dimensions n,m. As such we fill it with linear gradient
            // in both dimensions.
            for(int y = 0; y < n; y++)
                for(int x = 0; x < m; x++)
                    b[x + y*m] = x + y;

            printf("solving %ix%i matrix...\n", N, N);
            Conjugate_Gardient_Params large_params = params;
            large_params.max_iters = 100;
            TEST(matrix_conjugate_gradient_solve(A, x, b, N, &large_params), "must converge within max iters!");
            printf("done!\n");
            matrix_multiply(Ax, A, x, N, N, N, 1);
            TEST(are_near_scaled(Ax, b, N, large_params.tolerance*10));

            printf("inverting %ix%i matrix...\n", N, N);
            matrix_invert(Ainv, A, N);
            printf("done!\n");
            
            printf("Multiplying %ix%i matrix with its inverse...\n", N, N);
            matrix_multiply(I, A, Ainv, N, N, N, N);
            printf("done!\n");
            TEST(matrix_is_identity(I, 0.0001f, N));

            free(A);
            free(Ainv);
            free(I);
            free(b);
            free(x);
            free(Ax);
        }
    }
}