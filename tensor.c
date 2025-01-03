#include <stdlib.h>
#include <stdio.h>
#include "tensor.h"

TENSOR_ERR tensor_init(struct tensor *t, int *shape)
{
	void *data;
	int size = sizeof(double);
	int rank = 0;

	if (!t || !shape)
		return INVALID_ARGS;

	for (int i = 0; shape[i] != 0; i++) {
		if (shape[i] < 0)
			return INVALID_SHAPE;
		size *= shape[i];
		rank++;
	}

	data = malloc(size);
	if (!data)
		return INIT_ERROR;

	t->data = data;
	t->shape = shape;
	t->rank = rank;

	return SUCCESS;
}

TENSOR_ERR tensor_destroy(struct tensor *t)
{
	if (!t)
		return DESTR_ERROR;

	if (t->data)
		free(t->data);

	t->data = 0;
	t->rank = 0;
	t->shape = 0;

	return SUCCESS;
}
