#include "Basic.h"
#include<stdlib.h>
#include <stdio.h>
void checkStatus(culaStatus status)
{
	char buf[256];

	if (!status)
		return;

	culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
	printf("%s\n", buf);

	culaShutdown();
	system("pause");
	exit(EXIT_FAILURE);
}