/* tensor: main tensor structure.
 * @data: pointer to the start of the tensor.
 * @shape: Array of dimensions - terminated by a 0.
 * @rank: rank
 */

struct tensor {
	void *data;
	int *shape;
	int rank;
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
