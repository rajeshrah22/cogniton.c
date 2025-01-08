#include <string.h>

/* tensor: main tensor structure.
 * @data: pointer to the start of the tensor.
 * @shape: Array of dimensions - terminated by a 0.
 * @rank: rank
 */

struct tensor {
	double *data;
	int *shape;
	int rank;
	unsigned int memlen;
};

typedef enum tensor_err {
	SUCCESS,
	SHAPE_MISMATCH,
	INVALID_SHAPE,
	INVALID_ARGS,
	INIT_ERROR,
	DESTR_ERROR,
	IDK_SOMETHING_ELSE
} TENSOR_ERR;


TENSOR_ERR tensor_init(struct tensor *t, int *shape);
TENSOR_ERR tensor_destroy(struct tensor *t);
TENSOR_ERR tensor_copy(struct tensor *t, struct tensor *nt);
TENSOR_ERR tensor_multiply_constant(double m, struct tensor *t, struct tensor *res);
TENSOR_ERR tensor_elementwise_multiplication(struct tensor *a, struct tensor *b, struct tensor *res);
TENSOR_ERR tensor_add_constant(double c, struct tensor *t, struct tensor *res);
TENSOR_ERR tensor_addition(struct tensor *a, struct tensor *b, struct tensor *res);
TENSOR_ERR tensor_subtraction(struct tensor *a, struct tensor *b, struct tensor *res);
TENSOR_ERR tensor_elementwise_square(struct tensor *t, struct tensor *res);
TENSOR_ERR tensor_sum(struct tensor *t, double *sum);

static inline TENSOR_ERR check_same_shape(struct tensor *a, struct tensor *b)
{
	int cmp;

	if (a->rank != b->rank)
		return SHAPE_MISMATCH;

	if (!a->shape || !b->shape)
		return INVALID_ARGS;

	cmp = memcmp(a->shape, b->shape, a->rank * sizeof(int));
	if (cmp != 0)
		return SHAPE_MISMATCH;

	return SUCCESS;
}
