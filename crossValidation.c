void malloc_cpu(struct svm_problem *p)
{
	int i;
	p->l = prob.l;
	p->x = Malloc(struct svm_node, p->l);
	p->y = Malloc(double, p->l);
	for (i = 0; i < prob.l; i++)
	{
		(p->x + i) -> values = Malloc(double, prob.l + 1);
		(p->x + i) -> dim = prob.l + 1;
	}
	for (i = 0; i < prob.l; i++)
	{
		p->y[i] = prob.y[i];
	}
}

double crossValidation_cpu(struct svm_problem *p)
{
	double rate;
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double, prob.l);
	svm_cross_validation(p, &param, nr_fold, target);
	if (param.svm_type == EPSILON_SVR || param.svm_type == NU_SVR)
	{
		for (i = 0; i < prob.l; i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v - y) * (v - y);
			sumv += v;
			sumy += y;
			sumvv += v * v;
			sumyy += y * y;
			sumvy += v * y;
		}
		printf("cross validation mean squared error: %g\n", total_error / prob.l);
		printf("cross validation squared correlation coefficient: %g\n", ((prob.l * sumvy - sumv * sumy) * (prob.l * sumvy - sumv * sumy)) / ((prob.l * sumvv - sumv * sumv) * (prob.l * sumyy - sumy * sumy)));
	}
	else
	{
		for (i = 0; i < prob.l; i++)
		{
			if (target[i] == prob.y[i])
			{
				++total_correct;
			}
			rate = (100.0 * total_correct) / prob.l;
		}
	}
	free(target);
	return rate;
}

void kernalMatrixCalculation_gpu(struct svm_problem *p)
{
	double rate;
	kernalMatrix(p);
	param.kernel_type = PRECOMPUTED;
	rate = crossValidation_cpu(p);
	printf("cross validation: %g%%\n", rate);
}

void free_cpu(struct svm_problem *p)
{
	int i;
	for (i = 0; i < prob.l; i++)
	{
		free((p->x + i)->values);
	}
	free(p->x);
	free(p->y);
}

void crossValidation()
{
	struct svm_problem p;
	malloc_cpu(&p);
	kernalMatrixCalculation_gpu(&p);
	free_cpu(&p);
}
