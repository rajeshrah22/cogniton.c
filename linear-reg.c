#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"

struct lin_reg {
	double m;
	double c;
	struct lin_reg_hyper_params {
		double learn_rate;
		int iters;
	} hyp_params;
};

int forward_propagation(struct lin_reg *self, struct tensor *train_in, struct tensor *predictions)
{
	TENSOR_ERR err;

	if (!self || !train_in || !predictions)
		return -1;

	if (check_same_shape(train_in, predictions) != SUCCESS)
		return -1;

	err = tensor_multiply_constant(self->m, train_in, predictions);	
	if (err != SUCCESS)
		return -1;

	err = tensor_add_constant(self->c, predictions, predictions);	
	if (err != SUCCESS)
		return -1;

	return 0;
}

int cost_function(struct lin_reg *self, struct tensor *predictions, struct tensor *train_out, double *cost)
{
	TENSOR_ERR err;
	struct tensor pcopy;

	tensor_copy(predictions, &pcopy);

	if (!self || !predictions || !train_out)
		return -1;

	err = tensor_subtraction(predictions, train_out, &pcopy);
	if (err != SUCCESS)
		return -1;
	err = tensor_elementwise_square(&pcopy, &pcopy);
	if (err != SUCCESS)
		return -1;

	err = tensor_sum(&pcopy, cost);
	if (err != SUCCESS)
		return -1;

	err = tensor_destroy(&pcopy);
	if (err != SUCCESS)
		return -1;

	return 0;
}

int backward_propagation(struct lin_reg *self,
			 struct tensor *train_in,
			 struct tensor *train_out,
			 struct tensor *predictions)
{
	TENSOR_ERR err;
	double sum;
	double m_gradient;
	double c_gradient;
	struct tensor pcopy;

	// gradient in respect to c
	tensor_copy(predictions, &pcopy);

	err = tensor_subtraction(predictions, train_out, &pcopy);
	if (err != SUCCESS)
		return -1;

	err = tensor_sum(&pcopy, &sum);
	if (err != SUCCESS)
		return -1;

	sum = sum * 2 / train_out->shape[0];
	c_gradient = self->hyp_params.learn_rate * sum;

	self->c -= c_gradient;

	// gradient in respect to m
	tensor_copy(predictions, &pcopy);

	err = tensor_subtraction(predictions, train_out, &pcopy);
	if (err != SUCCESS)
		return -1;

	err = tensor_elementwise_multiplication(&pcopy, train_in, &pcopy);
	if (err != SUCCESS)
		return -1;

	err = tensor_sum(&pcopy, &sum);
	if (err != SUCCESS)
		return -1;

	sum = sum * 2 / train_out->shape[0];
	m_gradient = self->hyp_params.learn_rate * sum;

	self->m -= m_gradient;

	err = tensor_destroy(&pcopy);
	if (err != SUCCESS)
		return -1;

	return 0;
}

int train(struct lin_reg *self,
	  struct tensor *train_in,
	  struct tensor *train_out)
{
	TENSOR_ERR err;
	double cost;
	struct tensor predictions;
	tensor_copy(train_out, &predictions);

	for (int i = 0; i < self->hyp_params.iters; i++) {
		forward_propagation(self, train_in, &predictions);
		cost_function(self, &predictions, train_out, &cost);
		backward_propagation(self, train_in, train_out, &predictions);
		printf("Iteration %d, Cost: %f\n", i + 1, cost);
	}

	err = tensor_destroy(&predictions);
	if (err != SUCCESS)
		return -1;

	return 0;
}

int main(int argc, char **argv)
{
	struct lin_reg params;
	int shape[2] = {3, 0};
	struct tensor train_in;
	struct tensor train_out;
	TENSOR_ERR err;

	if (argc < 3) {
		printf("Usage: ./linear-reg [iters] [learn-rate]\n");
		return -1;
	}

	if (atoi(argv[1]) < 0) {
		printf("Iters should be positive\n");
		return -1;
	}

	// example training data
	double input[3] = {1, 2.1, 4};
	double output[3] = {1, 2.1, 3};

	err = tensor_init(&train_in, shape);
	if (err != SUCCESS) {
		perror("in tensor init failed\n");
		exit(2);
	}

	err = tensor_init(&train_out, shape);
	if (err != SUCCESS) {
		perror("out tensor init failed\n");
		exit(2);
	}

	memcpy(train_in.data, input, sizeof(input));
	memcpy(train_out.data, output, sizeof(output));

	params.hyp_params.iters = atoi(argv[1]);
	params.hyp_params.learn_rate = strtod(argv[2], 0);;
	params.m = 0.1;
	params.c = 0.1;

	train(&params, &train_in, &train_out);

	printf("m: %f\n", params.m);
	printf("c: %f\n", params.c);

	return 0;
}
