#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"

#define TEST_OK(num, msg) printf("ok %d - %s\n", num, msg)
#define TEST_NOT_OK(num, msg) printf("not ok %d - %s\n", num, msg)

int test_count = 0;

void test_tensor_init() {
	printf("# Testing tensor_init\n");
	struct tensor t;
	int shape[] = {2, 3, 0};

	test_count++;

	if (tensor_init(&t, shape) == SUCCESS) {
		if (t.rank == 2 && t.shape[0] == 2 && t.shape[1] == 3 && t.memlen == (2 * 3 * sizeof(double)) && t.data != NULL)
			TEST_OK(test_count, "tensor_init works with valid input");
		else
			TEST_NOT_OK(test_count, "tensor_init correctly allocates memory and sets attributes");
	} else {
		TEST_NOT_OK(test_count, "tensor_init with valid input");
	}

	tensor_destroy(&t);

	test_count++;

	if (tensor_init(NULL, shape) == INVALID_ARGS)
		TEST_OK(test_count, "tensor_init handles NULL tensor argument");
	else
		TEST_NOT_OK(test_count, "tensor_init handles NULL tensor argument");

	test_count++;

	if (tensor_init(&t, NULL) == INVALID_ARGS)
		TEST_OK(test_count, "tensor_init handles NULL shape argument");
	else
		TEST_NOT_OK(test_count, "tensor_init handles NULL shape argument");
}

void test_tensor_destroy() {
	printf("# Testing tensor_destroy\n");
	struct tensor t;
	int shape[] = {2, 3, 0};
	tensor_init(&t, shape);

	test_count++;

	if (tensor_destroy(&t) == SUCCESS && t.data == NULL && t.shape == NULL)
		TEST_OK(test_count, "tensor_destroy frees memory and resets attributes");
	else
		TEST_NOT_OK(test_count, "tensor_destroy frees memory and resets attributes");

	test_count++;

	if (tensor_destroy(NULL) == INVALID_ARGS)
		TEST_OK(test_count, "tensor_destroy handles NULL tensor argument");
	else
		TEST_NOT_OK(test_count, "tensor_destroy handles NULL tensor argument");
}

void test_tensor_copy() {
	printf("# Testing tensor_copy\n");
	struct tensor t, nt;
	int shape[] = {2, 2, 0};
	tensor_init(&t, shape);

	t.data[0] = 1.0;
	t.data[1] = 2.0;
	t.data[2] = 3.0;
	t.data[3] = 4.0;

	test_count++;
	if (tensor_copy(&t, &nt) == SUCCESS) {
		if (nt.rank == t.rank &&
		    nt.memlen == t.memlen &&
		    memcmp(nt.data, t.data, t.memlen) == 0 &&
		    memcmp(nt.shape, t.shape, t.rank * sizeof(int)) == 0)
			TEST_OK(test_count, "tensor_copy creates an identical copy of tensor");
		else
			TEST_NOT_OK(test_count, "tensor_copy creates an identical copy of tensor");

		tensor_destroy(&nt);
	} else {
		TEST_NOT_OK(test_count, "tensor_copy with valid input");
	}

	test_count++;

	if (tensor_copy(NULL, &nt) == INVALID_ARGS)
		TEST_OK(test_count, "tensor_copy handles NULL source tensor argument");
	else
		TEST_NOT_OK(test_count, "tensor_copy handles NULL source tensor argument");

	tensor_destroy(&t);
}

int main(void)
{
	test_tensor_init();
	test_tensor_destroy();
	test_tensor_copy();
}
