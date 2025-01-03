#include <stdio.h>
#include "tensor.h"

int main(void)
{
	struct tensor t;
	int shape[3] = {2, 3, 0};
	TENSOR_ERR err;

	err = tensor_init(&t, shape);
	if (err != SUCCESS) {
		perror("Something happened!\n");
	}

	if (t.rank != 2)
		printf("%d is not correct rank of 2\n", t.rank);

	tensor_destroy(&t);
}
