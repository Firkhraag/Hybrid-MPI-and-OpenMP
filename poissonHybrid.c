#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>
#include <sys/time.h>
#include <mpi.h>

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

// // Grid steps
// float xi(const int a1, const int localIndex, const int processOffset, const int stepX) {
//     // printf("wtf: %f\n", a1 + (localIndex + processOffset) * stepX);
//     // printf("wtf: %f\n", a1);
//     // printf("wtf: %f\n", localIndex);
//     // printf("wtf: %f\n", processOffset);
//     // printf("wtf: %f\n", stepX);
//     return a1 + (localIndex + processOffset) * stepX;
// }

// float yj(const int b1, const int localIndex, const int processOffset, const int stepY) {
//     return b1 + (localIndex + processOffset) * stepY;
// }

// // Grid elements
// float getGridElement(float* grid, const int i, const int j, const int n) {
//     return grid[i * (n + 1) + j];
// }

// // Difference schemes
// float leftDiffSchemeX(float* grid, const int i, const int j, const int n) {
//     return getGridElement(grid, i + 1, j, n) - getGridElement(grid, i, j, n);
// }

// float rightDiffSchemeX(float* grid, const int i, const int j, const int n) {
//     return getGridElement(grid, i, j, n) - getGridElement(grid, i - 1, j, n);
// }

// float leftDiffSchemeY(float* grid, const int i, const int j, const int n) {
//     return getGridElement(grid, i, j + 1, n) - getGridElement(grid, i, j, n);
// }

// float rightDiffSchemeY(float* grid, const int i, const int j, const int n) {
//     return getGridElement(grid, i, j, n) - getGridElement(grid, i, j - 1, n);
// }

// // Difference schemes
// float leftDiffSchemeX(float* grid, const int i, const int j, const int blockWidth) {
//     const int index = i * blockWidth + j;
//     return grid[index + blockWidth] - grid[index];
// }

// float rightDiffSchemeX(float* grid, const int i, const int j, const int blockWidth) {
//     const int index = i * blockWidth + j;
//     return grid[index] - grid[index - blockWidth];
// }

// float leftDiffSchemeY(float* grid, const int i, const int j, const int blockWidth) {
//     const int index = i * blockWidth + j;
//     return grid[index + 1] - grid[index];
// }

// float rightDiffSchemeY(float* grid, const int i, const int j, const int blockWidth) {
//     const int index = i * blockWidth + j;
//     return grid[index] - grid[index - 1];
// }

// Laplace difference scheme
float laplaceDiffScheme(float* grid, const float x, const float y, const int index, const int stepX, const int stepY,
                        const int stepXCoeff, const int stepYCoeff, const int blockWidth) {

    return -(stepXCoeff * (k(x + 0.5 * stepX) * (grid[index + blockWidth] - grid[index]) -
        k(x - 0.5 * stepX) * (grid[index] - grid[index - blockWidth])) +
        stepYCoeff * k(x) * ((grid[index + 1] - grid[index]) - (grid[index] - grid[index - 1]))) +
        q(x, y) * grid[index];
}

// Dot product
float dotProduct(float* grid1, float* grid2, int blockWidth, int blockHeight, float stepX, float stepY) {
    float result = 0;
    #pragma omp parallel for schedule (static) reduction(+:result)
    for (int i = 1; i < blockHeight - 1; i++) {
        for (int j = 1; j < blockWidth - 1; j++) {
            const int index = i * blockWidth + j;
            result += grid1[index] * grid2[index];
        }
    }
    result = result * stepX * stepY;

    float sum;
    // Gathers to root and reduce with sum: send_data, recv_data, count, datatype, op, root, communicator
    MPI_Reduce(&result, &sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    // Broadcasts from root to other processes: buffer, count, datatype, root, communicator
    MPI_Bcast(&sum, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    return sum;
}

// float dotProductMPI(float var, const int currentRank, const int size) {
//     float* sums;
//     if (currentRank == 0){
//         float* sums = (float*)malloc(size * sizeof(float));
//     }

//     // Gather to root: sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm
//     MPI_Gather(&var, 1, MPI_FLOAT, sums, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

//     float sum = 0;
//     if (currentRank == 0) {
//         // Sum all sums
//         #pragma omp parallel for schedule (static) reduction(+:sum)
//         for (int i = 0; i < size; i++)
//             sum += sums[i];
//         free(sums);
//     }

//     MPI_Bcast(&sum, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
//     return sum;
// }

// float dotProductMPI(float var, const int currentRank, const int size) {
//     float sum;
//     // Gathers to root and reduce with sum: send_data, recv_data, count, datatype, op, root, communicator
//     MPI_Reduce(&var, &sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
//     // Broadcasts from root to other processes: buffer, count, datatype, root, communicator
//     MPI_Bcast(&sum, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
//     return sum;
// }

// For each block
void passInformationBetweenProcesses(const int currentRank, const int numOfBlocksX, const int numOfBlocksY, const int blockPositionX, const int blockPositionY, float* grid,
                                    const int blockWidth, const int blockHeight) {

    bool up = true;
    if (blockPositionX == 0) {
        up = false;
    }
    bool bottom = true;
    if (blockPositionX == numOfBlocksX - 1) {
        bottom = false;
    }
    bool left = true;
    if (blockPositionY == 0) {
        left = false;
    }
    bool right = true;
    if (blockPositionX == numOfBlocksY - 1) {
        right = false;
    }

    float* sendUp;
    float* sendBottom;
    float* sendLeft;
    float* sendRight;

    /*
        Process grid
        - - - - - - - -
        - * * * * * * -
        - * + + + + * -
        - * + + + + * -
        - * + + + + * -
        - * + + + + * -
        - * * * * * * -
        - - - - - - - -

        - points that we receive from other processes
        * points that we send to other processes
        + internal points
    */
    const int width  = blockWidth - 2;
    const int height = blockHeight - 2;

    const int upperNeighborRank = numOfBlocksY * (blockPositionX - 1) + blockPositionY;
    const int bottomNeighborRank = numOfBlocksY * (blockPositionX + 1) + blockPositionY;
    const int leftNeighborRank = numOfBlocksY * blockPositionX + blockPositionY - 1;
    const int rightNeighborRank = numOfBlocksY * blockPositionX + blockPositionY + 1;

    MPI_Status status;
    MPI_Request leftSendRequest, rightSendRequest, upSendRequest, bottomSendRequest;

    int i;
    // Sending nodes near boundary to other processes
    #pragma omp parallel sections private(i)
    {
        #pragma omp section
        if (up) {
            sendUp = (float*)malloc(width * sizeof(float));
            for (i = 0; i < width; i++) {
                sendUp[i] = grid[blockWidth + (i + 1)];
            }
            // Nonblocking send: buf, count, datatype, destination, tag, communicator, request
            MPI_Isend(sendUp, width, MPI_FLOAT, upperNeighborRank, 0, MPI_COMM_WORLD, &upSendRequest);
	    }
        #pragma omp section
        if (bottom) {
            sendBottom = (float*)malloc(width * sizeof(float));
            for (i = 0; i < width; i++) {
                sendBottom[i] = grid[height * blockWidth + (i + 1)];
            }
            MPI_Isend(sendBottom, width, MPI_FLOAT, bottomNeighborRank, 0, MPI_COMM_WORLD, &bottomSendRequest);
	    }
        #pragma omp section
        if (left) {
            sendLeft = (float*)malloc(height * sizeof(float));
            for (i = 0; i < height; i++) {
                sendLeft[i] = grid[(i + 1) * blockWidth + 1];
            }
            MPI_Isend(sendLeft, height, MPI_FLOAT, leftNeighborRank, 0, MPI_COMM_WORLD, &leftSendRequest);
	    }
        #pragma omp section
        if (right) {
            sendRight = (float*)malloc(height * sizeof(float));
            for (i = 0; i < height; i++) {
                sendRight[i] = grid[(i + 1) * blockWidth + width];
            }
            MPI_Isend(sendRight, height, MPI_FLOAT, rightNeighborRank, 0, MPI_COMM_WORLD, &rightSendRequest);
	    }
    }

    // Receive boundary nodes from other processes
    #pragma omp parallel sections private(i)
    {
        #pragma omp section
        if (up) {
            float* receiveUp = (float*)malloc(width * sizeof(float));
            // Blocking receive: buf, count, datatype, source, tag, communicator, status
            MPI_Recv(receiveUp, width, MPI_FLOAT, upperNeighborRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (i = 0; i < width; i++) {
                grid[i + 1] = receiveUp[i];
            }
            free(receiveUp);
        }
        #pragma omp section
        if (bottom) {
            float* receiveBottom = (float*)malloc(width * sizeof(float));
            MPI_Recv(receiveBottom, width, MPI_FLOAT, bottomNeighborRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (i = 0; i < width; i++) {
                grid[(height + 1) * blockWidth + (i + 1)] = receiveBottom[i];
            }
            free(receiveBottom);
        }
        #pragma omp section
        if (left) {
            float* receiveLeft = (float*)malloc(height * sizeof(float));
            MPI_Recv(receiveLeft, height, MPI_FLOAT, bottomNeighborRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (i = 0; i < height; i++) {
                grid[(i + 1) * blockWidth] = receiveLeft[i];
            }
            free(receiveLeft);
        }
        #pragma omp section
        if (right) {
            float* receiveRight = (float*)malloc(height * sizeof(float));
            MPI_Recv(receiveRight, height, MPI_FLOAT, bottomNeighborRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (i = 0; i < height; i++) {
                grid[(i + 1) * blockWidth + (width + 1)] = receiveRight[i];
            }
            free(receiveRight);
        }
    }

    // Wait sending to compelete and delete allocated arrays
    if (up) {
        // Wait for request completion
        MPI_Wait(&upSendRequest, &status);
        free(sendUp);
    }
    if (bottom) {
        MPI_Wait(&bottomSendRequest, &status);
        free(sendBottom);
    }
    if (left) {
        MPI_Wait(&leftSendRequest, &status);
        free(sendLeft);
    }
    if (right) {
        MPI_Wait(&rightSendRequest, &status);
        free(sendRight);
    }
}


int main(int argc, char **argv) {

    struct timeval start, end;

    gettimeofday(&start, NULL);

    // Rectangle
    const float a1 = -1.0;
    const float a2 = 2.0;
    const float b1 = -2.0;
    const float b2 = 2.0;

    // Epsilon
    const float eps = 1e-3;

    // Square grid
    const int n = 10;
    // From 0 to n
    // const int dim = (n + 1) * (n + 1);

    // Step
    const float stepX = (a2 - a1) / n;
    const float stepY = (b2 - b1) / n;

    const float stepXCoeff = 1 / (stepX * stepX);
    const float stepYCoeff = 1 / (stepY * stepY);

    // // Grid with i, j as an array
    // float* grid = (float*)malloc((m + 1) * (n + 1) * sizeof(float));

    // // Real values for comparison
    // float* realValues = (float*)malloc((m + 1) * (n + 1) * sizeof(float));

    // float **realValues;
    // realValues = (float **)malloc((m - 1) * sizeof(float *));
    // for (int i = 0; i < (m - 1); i++) {
    //     *(realValues + i) = (float *)malloc((n - 1) * sizeof(float));
    // }

    float tau;

	int size;
	int currentRank;

	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &currentRank);

    // float starttime = MPI_Wtime();

    int numOfBlocksY = 1;
	while (size > 2 * numOfBlocksY * numOfBlocksY) {
		numOfBlocksY *= 2;
	}
	const int numOfBlocksX = size / numOfBlocksY;

	// Global block position
    const int blockPositionX = currentRank / numOfBlocksY;
    const int blockPositionY = currentRank % numOfBlocksY;

	const int blockSizeX = (n - 1) / numOfBlocksX;
	const int blockSizeY = (n - 1) / numOfBlocksY;

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
    printf("------\n");

    // Local grid approximation
    float* grid = (float*)malloc(blockWidth * blockHeight * sizeof(float));

    // Difference between two steps
    float* gridDiff = (float*)malloc(blockWidth * blockHeight * sizeof(float));

    // Local real values
    float* realValues = (float*)malloc(blockWidth * blockHeight * sizeof(float));

    // Local residuals
    float* rk = (float*)malloc(blockWidth * blockHeight * sizeof(float));

    // A * rk
    float* ark = (float*)malloc(blockWidth * blockHeight * sizeof(float));

    // // Error
    // float* error = (float*)malloc(blockWidth * blockHeight * sizeof(float));

	int i;
	int j;
    float stopCondition;

    // Find real values in the block
	#pragma omp parallel for schedule (static)
	for (i = 1; i < blockHeight - 1; i++) {
		for (j = 1; j < blockWidth - 1; j++) {
			realValues[i * blockWidth + j] = u(a1 + (i + startX) * stepX, b1 + (j + startY) * stepY);
		}
	}

    // Find global boundary values in the block
	if (startX == 0) {
        #pragma omp parallel for schedule (static)
		for (j = 0; j < blockWidth; j++) {
			grid[j] = u(a1 + startX * stepX, b1 + (j + startY) * stepY);
		}
	}

	if (endX == n) {
        #pragma omp parallel for schedule (static)
		for (j = 0; j < blockWidth; j++) {
            grid[(blockHeight - 1) * blockWidth + j] = u(a1 + (blockHeight - 1 + startX) * stepX, b1 + (j + startY) * stepY);
		}
	}

	if (startY == 0) {
        #pragma omp parallel for schedule (static)
		for (i = 0; i < blockHeight; i++) {
            grid[i * blockWidth] = u(a1 + (i + startX) * stepX, b1 + startY * stepY);
		}
	}

	if (endY == n) {
        #pragma omp parallel for schedule (static)
		for (i = 0; i < blockHeight; i++) {
            grid[i * blockWidth + (blockWidth - 1)] = u(a1 + (i + startX) * stepX, b1 + (blockWidth - 1 + startY) * stepY);
		}
    }

    // Initializing grid with starting values
    #pragma omp parallel for schedule (static)
    for (i = 1; i < blockHeight - 1; i++) {
		for (j = 1; j < blockWidth - 1; j++) {
			grid[i * blockWidth + j] = 2;       // CHANGE TO 0!!!
		}
	}

    do {
        // Find residual
        #pragma omp parallel for schedule (static)
        for (i = 1; i < blockHeight - 1; i++) {
            for (j = 1; j < blockWidth - 1; j++) {
                const float x = a1 + (i + startX) * stepX;
                const float y = b1 + (j + startY) * stepY;
                const int index = i * blockWidth + j;
                rk[i * blockWidth + j] =
                    laplaceDiffScheme(grid, x, y, index, stepX, stepY, stepXCoeff, stepYCoeff, blockWidth) - F(x, y);
            }
        }

        // Pass residuals to adjacent processes
        passInformationBetweenProcesses(currentRank, numOfBlocksX, numOfBlocksY, blockPositionX, blockPositionY, rk, blockWidth, blockHeight);

        // Find A * rk
        #pragma omp parallel for schedule (static)
        for (i = 1; i < blockHeight - 1; i++) {
            for (j = 1; j < blockWidth - 1; j++) {
                const float x = a1 + (i + startX) * stepX;
                const float y = b1 + (j + startY) * stepY;
                const int index = i * blockWidth + j;
                ark[index] = laplaceDiffScheme(rk, x, y, index, stepX, stepY, stepXCoeff, stepYCoeff, blockWidth);
                printf("Values: %f\n", ark[index]);
            }
        }

        // Find tau
        float tau1 = dotProduct(ark, rk, blockWidth, blockHeight, stepX, stepY);
        float tau2 = dotProduct(ark, ark, blockWidth, blockHeight, stepX, stepY);

        tau = tau1 / tau2;

        // Find new approximation
        #pragma omp parallel for schedule (static)
        for (i = 1; i < blockHeight - 1; i++) {
            for (j = 1; j < blockWidth - 1; j++) {
                const int index = i * blockWidth + j;
                const float gridElementOld = grid[index];
                grid[index] -= tau * rk[index];
                gridDiff[index] = grid[index] - gridElementOld;
            }
        }

        // // Deviation
        // #pragma omp parallel for schedule (static)
        // for (i = 1; i < blockHeight - 1; i++) {
        //     for (j = 1; j < blockWidth - 1; j++) {
        //         const int index = i * blockWidth + j;
        //         error[index] = (realValues[index] - grid[index]) * (realValues[index] - grid[index]);
        //     }
        // }

        stopCondition = dotProduct(gridDiff, gridDiff, blockWidth, blockHeight, stepX, stepY);

        // Wait for all processes to complete the step
        MPI_Barrier(MPI_COMM_WORLD);

    } while (stopCondition > eps);

    // Deviation
    float localError = 0;
    #pragma omp parallel for schedule (static) reduction(+:localError)
    for (i = 1; i < blockHeight - 1; i++) {
        for (j = 1; j < blockWidth - 1; j++) {
            const int index = i * blockWidth + j;
            localError += (realValues[index] - grid[index]) * (realValues[index] - grid[index]);
        }
    }

    float error;
    // Gathers to root and reduce with sum: send_data, recv_data, count, datatype, op, root, communicator
    MPI_Reduce(&localError, &error, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    free(grid);
    free(gridDiff);
    free(realValues);
    free(rk);
    free(ark);

    // float endtime = MPI_Wtime();
    // printf("Execution time: %f sec\n", endtime-starttime);

    MPI_Finalize();

    FILE *f = fopen("resultTest.txt", "w");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    gettimeofday(&end, NULL);
    double time_taken = end.tv_sec + end.tv_usec / 1e6 -
                        start.tv_sec - start.tv_usec / 1e6; // in seconds
    fprintf(f, "Error: %f\n", error);
    fprintf(f, "Execution time: %f\n", time_taken);
    fclose(f);
    return 0;
}
