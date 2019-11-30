#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>
#include <sys/time.h>
// #include <mpi.h>

// Bool type
typedef enum {false, true} bool;

// Given functions
float u(float x, float y) {
    return exp(1 - (x + y) * (x + y));
}

float k(float x) {
    return 4 + x;
}

float q(float x, float y) {
    return (x + y) * (x + y);
}

float F(float x, float y) {
    return u(x, y) * ((x + y) * (x + y + 2) - (4 + x) * (-4 + 8 * (x + y) * (x + y)));
}

// Dot product
float dotProduct(float* grid1, float* grid2, int blockWidth, int blockHeight, float stepX, float stepY) {
    float result = 0;
    #pragma omp parallel for reduction(+:result)
    for (int i = 1; i < blockHeight - 1; i++) {
        for (int j = 1; j < blockWidth - 1; j++) {
            const int index = i * blockWidth + j;
            result += grid1[index] * grid2[index];
        }
    }
    result = result * stepX * stepY;

    return result;
}

int main(int argc, char **argv) {

    struct timeval start, end;

    // Get start time
    gettimeofday(&start, NULL);

    // Rectangle
    const float a1 = -1.0;
    const float a2 = 2.0;
    const float b1 = -2.0;
    const float b2 = 2.0;

    // Epsilon
    const float eps = 1e-5;

    // Square grid
    const int n = 10;

    // Step
    const float stepX = (a2 - a1) / n;
    const float stepY = (b2 - b1) / n;

    const float stepXCoeff = 1 / (stepX * stepX);
    const float stepYCoeff = 1 / (stepY * stepY);

    float tau;

	int size = 1;
	int currentRank = 0;

    int numOfBlocksY = 1;
	while (size > 2 * numOfBlocksY * numOfBlocksY) {
		numOfBlocksY *= 2;
	}
	const int numOfBlocksX = size / numOfBlocksY;

	// Global block position
    const int blockPositionX = currentRank / numOfBlocksY;
    const int blockPositionY = currentRank % numOfBlocksY;

	const int blockSizeX = n / numOfBlocksX;
	const int blockSizeY = n / numOfBlocksY;

	const int startX = fmax(0, blockSizeX * blockPositionX - 1);
	const int endX = blockPositionX + 1 < numOfBlocksX ? startX + blockSizeX : n;

	const int startY = fmax(0, blockSizeY * blockPositionY - 1);
	const int endY = blockPositionY + 1 < numOfBlocksY ? startY + blockSizeY : n;

	const int blockHeight = endX - startX + 1;
	const int blockWidth = endY - startY + 1;

    printf("------\n");
    printf("Size: %d\n", size);
    printf("Rank: %d\n", currentRank);
    printf("NumOfBlocksY: %d\n", numOfBlocksY);
    printf("NumOfBlocksX: %d\n", numOfBlocksX);
    printf("BlockPositionX: %d\n", blockPositionX);
    printf("BlockPositionY: %d\n", blockPositionY);
    printf("BlockSizeX: %d\n", blockSizeX);
    printf("BlockSizeY: %d\n", blockSizeY);
    printf("StartX: %d\n", startX);
    printf("EndX: %d\n", endX);
    printf("StartY: %d\n", startY);
    printf("EndY: %d\n", endY);
    printf("BlockHeight: %d\n", blockHeight);
    printf("BlockWidth: %d\n", blockWidth);
    printf("------\n");

    // Local grid approximation array
    float* grid = (float*)malloc(blockWidth * blockHeight * sizeof(float));

    // Difference between two steps array
    float* gridDiff = (float*)malloc(blockWidth * blockHeight * sizeof(float));

    // Local real values array
    float* realValues = (float*)malloc(blockWidth * blockHeight * sizeof(float));

    // Local residuals array
    float* rk = (float*)malloc(blockWidth * blockHeight * sizeof(float));

    // A * rk array
    float* ark = (float*)malloc(blockWidth * blockHeight * sizeof(float));

    float stopCondition;

    // Find real values in the block
	#pragma omp parallel for
	for (int i = 1; i < blockHeight - 1; i++) {
		for (int j = 1; j < blockWidth - 1; j++) {
			realValues[i * blockWidth + j] = u(a1 + (i + startX) * stepX, b1 + (j + startY) * stepY);
		}
	}

    // Find global boundary values in the block
	if (startX == 0) {
        #pragma omp parallel for
		for (int j = 0; j < blockWidth; j++) {
			grid[j] = u(a1 + startX * stepX, b1 + (j + startY) * stepY);
		}
	}

	if (endX == n) {
        #pragma omp parallel for
		for (int j = 0; j < blockWidth; j++) {
            grid[(blockHeight - 1) * blockWidth + j] = u(a1 + (blockHeight - 1 + startX) * stepX, b1 + (j + startY) * stepY);
		}
	}

	if (startY == 0) {
        #pragma omp parallel for
		for (int i = 0; i < blockHeight; i++) {
            grid[i * blockWidth] = u(a1 + (i + startX) * stepX, b1 + startY * stepY);
		}
	}

	if (endY == n) {
        #pragma omp parallel for
		for (int i = 0; i < blockHeight; i++) {
            grid[i * blockWidth + (blockWidth - 1)] = u(a1 + (i + startX) * stepX, b1 + (blockWidth - 1 + startY) * stepY);
		}
    }

    // Initializing grid with starting values
    #pragma omp parallel for
    for (int i = 1; i < blockHeight - 1; i++) {
		for (int j = 1; j < blockWidth - 1; j++) {
			grid[i * blockWidth + j] = 0;
		}
	}

    FILE *f = fopen("test2.txt", "w");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    int step = -1;
    float error;
    do {
        step++;
        // Find residual using difference scheme
        #pragma omp parallel for
        for (int i = 1; i < blockHeight - 1; i++) {
            for (int j = 1; j < blockWidth - 1; j++) {
                const float x = a1 + (i + startX) * stepX;
                const float y = b1 + (j + startY) * stepY;
                const int index = i * blockWidth + j;
                rk[index] = -(
                    stepXCoeff * (k(x + 0.5 * stepX) * (grid[index + blockWidth] - grid[index]) -
                    k(x - 0.5 * stepX) * (grid[index] - grid[index - blockWidth])) +
                    stepYCoeff * k(x) * ((grid[index + 1] - grid[index]) -
                    (grid[index] - grid[index - 1]))) +
                    q(x, y) * grid[index] - F(x, y);
                printf("Found: %f\n", rk[index]);
            }
        }

        // Find A * rk using difference scheme
        #pragma omp parallel for
        for (int i = 1; i < blockHeight - 1; i++) {
            for (int j = 1; j < blockWidth - 1; j++) {
                const float x = a1 + (i + startX) * stepX;
                const float y = b1 + (j + startY) * stepY;
                const int index = i * blockWidth + j;
                ark[index] = -(
                    stepXCoeff * (k(x + 0.5 * stepX) * (rk[index + blockWidth] - rk[index]) -
                    k(x - 0.5 * stepX) * (rk[index] - rk[index - blockWidth])) +
                    stepYCoeff * k(x) * ((rk[index + 1] - rk[index]) -
                    (rk[index] - rk[index - 1]))) +
                    q(x, y) * rk[index];
            }
        }

        // Find tau
        float tau1 = dotProduct(ark, rk, blockWidth, blockHeight, stepX, stepY);
        float tau2 = dotProduct(ark, ark, blockWidth, blockHeight, stepX, stepY);

        tau = tau1 / tau2;

        // Find new approximation
        #pragma omp parallel for
        for (int i = 1; i < blockHeight - 1; i++) {
            for (int j = 1; j < blockWidth - 1; j++) {
                const int index = i * blockWidth + j;
                const float gridElementOld = grid[index];
                grid[index] -= tau * rk[index];
                gridDiff[index] = grid[index] - gridElementOld;
            }
        }

        // Deviation
        error = 0;
        #pragma omp parallel for reduction(+:error)
        for (int i = 1; i < blockHeight - 1; i++) {
            for (int j = 1; j < blockWidth - 1; j++) {
                const int index = i * blockWidth + j;
                error += (realValues[index] - grid[index]) * (realValues[index] - grid[index]);
            }
        }

        fprintf(f, "Step: %d. Error: %f\n", step, error);

        stopCondition = sqrt(dotProduct(gridDiff, gridDiff, blockWidth, blockHeight, stepX, stepY));
        printf("Stop: %f", stopCondition);
    } while (stopCondition > eps);

    free(gridDiff);
    free(rk);
    free(ark);

    // End time
    gettimeofday(&end, NULL);
    // In seconds
    double time_taken = end.tv_sec + end.tv_usec / 1e6 -
                        start.tv_sec - start.tv_usec / 1e6;
    fprintf(f, "Execution time: %f\n", time_taken);

    // for (int i = 1; i < blockHeight - 1; i++) {
    //     for (int j = 1; j < blockWidth - 1; j++) {
    //         fprintf(f, "Original function: %f, Approximated result: %f\n", realValues[i * blockWidth + j], grid[i * blockWidth + j]);
    //     }
    // }

    fclose(f);
    free(realValues);
    free(grid);


    return 0;
}
