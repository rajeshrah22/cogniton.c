#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "tensor.h"

TENSOR_ERR tensor_init(struct tensor *t, int *shape)
{
	double *data;
	int size = sizeof(double);
	int rank = 0;
	int *res;

	if (!t || !shape)
		return INVALID_ARGS;

	for (int i = 0; shape[i] != 0; i++) {
		if (shape[i] < 0)
			return INVALID_SHAPE;
		size *= shape[i];
		rank++;
	}

	if (rank == 0)
		return INVALID_ARGS;

	t->shape = (int *)malloc((rank + 1) * sizeof(int));
	if (!t->shape)
		return INIT_ERROR;

	res = memcpy(t->shape, shape, (rank + 1) * sizeof(int));
	if (!res)
		goto out_free_shape;

	data = (double *)malloc(size);
	if (!data)
		goto out_free_shape;

	t->data = data;
	t->rank = rank;
	t->memlen = size;

	return SUCCESS;
out_free_shape:
	free(t->shape);
	t->shape = 0;
	return INIT_ERROR;
}

TENSOR_ERR tensor_destroy(struct tensor *t)
{
	if (!t)
		return INVALID_ARGS;

	if (t->data)
		free(t->data);
	if (t->shape)
		free(t->shape);

	t->data = 0;
	t->rank = 0;
	t->shape = 0;
	t->memlen = 0;

	return SUCCESS;
}

TENSOR_ERR tensor_copy(struct tensor *t, struct tensor *nt)
{
	void *res;

	if (!t || !t->data || !t->rank || !t->shape || !t->memlen)
		return INVALID_ARGS;

	nt->data = (double *)malloc(t->memlen);
	if (!nt->data)
		return IDK_SOMETHING_ELSE;

	res = memcpy(nt->data, t->data, t->memlen);
	if (!res || res != nt->data)
		goto out_free_data;

	nt->shape = (int *)malloc((t->rank + 1) * sizeof(int));
	if (!nt->shape)
		return IDK_SOMETHING_ELSE;

	res = memcpy(nt->shape, t->shape, (t->rank + 1) * sizeof(int));
	if (!res)
		goto out_free_shape;

	nt->rank = t->rank;
	nt->memlen = t->memlen;

	return SUCCESS;
out_free_shape:
	free(nt->shape);
	nt->shape = 0;
out_free_data:
	free(nt->data);
	nt->data = 0;
	return IDK_SOMETHING_ELSE;
}

TENSOR_ERR tensor_multiply_constant(double c, struct tensor *t, struct tensor *res)
{
	if (!t || !t->data || !res || !res->data)
		return INVALID_ARGS;
	if (check_same_shape(t, res) != SUCCESS)
		return SHAPE_MISMATCH;

	for (unsigned int i = 0; i < t->memlen / sizeof(double); i++)
		res->data[i] = t->data[i] * c;

	return SUCCESS;
}

TENSOR_ERR tensor_elementwise_multiplication(struct tensor *a, struct tensor *b, struct tensor *res)
{
	if (!a || !b || !res || !a->data || !b->data || !res->data)
		return INVALID_ARGS;
	if (check_same_shape(a, b) != SUCCESS && check_same_shape(a, res) != SUCCESS)
		return SHAPE_MISMATCH;

	for (unsigned int i = 0; i < a->memlen / sizeof(double); i++)
		res->data[i] = a->data[i] * b->data[i];

	return SUCCESS;
}

TENSOR_ERR tensor_add_constant(double c, struct tensor *t, struct tensor *res)
{
	if (!t || !t->data || !res || !res->data)
		return INVALID_ARGS;
	if (check_same_shape(t, res) != SUCCESS)
		return SHAPE_MISMATCH;

	for (unsigned int i = 0; i < t->memlen / sizeof(double); i++)
		res->data[i] = t->data[i] + c;

	return SUCCESS;
}

TENSOR_ERR tensor_addition(struct tensor *a, struct tensor *b, struct tensor *res)
{
	if (!a || !b || !res || !a->data || !b->data || !res->data)
		return INVALID_ARGS;
	if (check_same_shape(a, b) != SUCCESS && check_same_shape(a, res) != SUCCESS)
		return SHAPE_MISMATCH;

	for (unsigned int i = 0; i < a->memlen / sizeof(double); i++)
		res->data[i] = a->data[i] + b->data[i];

	return SUCCESS;
}

TENSOR_ERR tensor_subtraction(struct tensor *a, struct tensor *b, struct tensor *res)
{
	if (!a || !b || !res || !a->data || !b->data || !res->data)
		return INVALID_ARGS;
	if (check_same_shape(a, b) != SUCCESS && check_same_shape(a, res) != SUCCESS)
		return SHAPE_MISMATCH;

	for (unsigned int i = 0; i < a->memlen / sizeof(double); i++)
		res->data[i] = a->data[i] - b->data[i];

	return SUCCESS;
}

TENSOR_ERR tensor_elementwise_square(struct tensor *t, struct tensor *res)
{

	if (!t || !t->data || !res || !res->data)
		return INVALID_ARGS;
	if (check_same_shape(t, res) != SUCCESS)
		return SHAPE_MISMATCH;

	for (unsigned int i = 0; i < t->memlen / sizeof(double); i++)
		res->data[i] = t->data[i] * t->data[i];

	return SUCCESS;
}

TENSOR_ERR tensor_sum(struct tensor *t, double *sum)
{
	int _sum = 0;

	if (!t || !t->data || !sum)
		return INVALID_ARGS;

	for (unsigned int i = 0; i < t->memlen / sizeof(double); i++)
		_sum += t->data[i];

	*sum = _sum;

	return SUCCESS;
}
